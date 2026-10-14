"""Batched observation-only cash PPO with iterative snapshots and holdout evaluation."""

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from zimortal.engine import RuleClarificationRequired
from zimortal.model.encoding import encode_action, encode_observation

from . import decision_audit
from .active import atomic_json
from .cashq import digest
from .rollout_engine import RolloutEngine
from .runtime import load_model, save_model


def pack(features, actions, device):
    width = max(map(len, actions))
    padded = np.zeros((len(actions), width, actions[0].shape[-1]), np.float32)
    mask = np.zeros((len(actions), width), bool)
    for i, row in enumerate(actions):
        padded[i, : len(row)] = row
        mask[i, : len(row)] = True
    return (
        torch.from_numpy(np.stack(features)).to(device),
        torch.from_numpy(padded).to(device),
        torch.from_numpy(mask).to(device),
    )


def infer(model, observations, device, generator=None, details=None, temperature=1.0):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("sampling temperature must be finite and positive")
    features = [encode_observation(obs, model.feature_version) for obs in observations]
    actions = [
        np.stack([encode_action(a, obs.player) for a in obs.legal_actions]) for obs in observations
    ]
    with torch.inference_mode():
        logits, values, _ = model(*pack(features, actions, device))
        logp = (logits / temperature).log_softmax(-1)
        indices = (
            torch.multinomial(logp.exp(), 1, generator=generator).squeeze(-1)
            if generator is not None
            else logits.argmax(-1)
        )
        chosen_logp = logp.gather(1, indices[:, None]).squeeze(1)
        if details is not None:
            details["probabilities"] = logp.exp().cpu().tolist()
    return (
        indices.cpu().tolist(),
        chosen_logp.cpu().tolist(),
        values.cpu().tolist(),
        features,
        actions,
    )


def play(
    model,
    opponents,
    specs,
    *,
    device="cpu",
    slots=32,
    seed=0,
    training=False,
    audit=None,
    audit_parent=None,
    sampling_temperature=1.0,
    replay_sink=None,
    decision_recorder=None,
    decision_sink=None,
):
    """Each spec is seed, dealer, learner seat, opponent index (-1=self-play).

    Complete games are committed together; undefined-rule games contribute no
    rows or fabricated zero rewards. Engine states never enter neural inputs.
    """
    if decision_recorder is not None and (decision_sink is None or audit is not None):
        raise ValueError("decision recorder requires sink and separate legacy audit")
    engine = RolloutEngine()
    generator = torch.Generator(device=device).manual_seed(seed) if training else None
    model.eval()
    for opponent in opponents:
        opponent.eval()
    active, cursor, rows, results, questions = {}, 0, [], [], []

    def finish(index, state, spec, trajectories):
        state.validate()
        # Full replay is independently checked with the ordinary public engine.
        from zimortal.engine import RuleEngine

        replay_engine = RuleEngine()
        initial = replay_engine.new_game(spec[0], spec[1])
        if replay_engine.replay(initial, state.history).serialize() != state.serialize():
            raise RuntimeError("RL terminal replay mismatch")
        if replay_sink is not None:
            replay_sink(
                {"seed": spec[0], "dealer": spec[1], "seat": spec[2], "state": state.serialize()}
            )
        payments = state.settlement.payments if state.settlement else (0, 0, 0)
        if audit is not None:
            audit.extend(
                {**r, "terminal_cash": payments[r["player"]], "winner": state.winner}
                for r in active[index][4]
            )
        if decision_sink is not None:
            decision_sink([{**r, "terminal_cash": payments[spec[2]]} for r in active[index][4]])
        for seat, trajectory in enumerate(trajectories):
            rows.extend((*row, payments[seat] / 100.0) for row in trajectory)
        results.append(
            {
                "seed": spec[0],
                "seat": spec[2],
                "opponent": spec[3],
                "cash": payments[spec[2]],
                "winner": state.winner,
                "steps": len(state.history),
                "decisions": sum(map(len, trajectories)),
                "history_sha256": hashlib.sha256(
                    json.dumps(json.loads(state.serialize())["history"], sort_keys=True).encode()
                ).hexdigest(),
            }
        )
        del active[index]

    while cursor < len(specs) or active:
        for index in range(slots):
            if index not in active and cursor < len(specs):
                spec = specs[cursor]
                active[index] = [engine.new_game(spec[0], spec[1]), spec, [[], [], []], 0, []]
                cursor += 1
        waiting = {}
        for index, (state, spec, trajectories, steps, _) in list(active.items()):
            if state.terminal:
                finish(index, state, spec, trajectories)
                continue
            if steps >= 1000:
                raise RuntimeError("RL game exceeded step limit")
            try:
                acts = engine.legal_actions(state)
                actor = acts[0].player
                if len(acts) == 1:
                    active[index][0] = engine.step(state, acts[0])
                    active[index][3] += 1
                    continue
                obs = engine.observation(state, actor)
                owner = -1 if actor == spec[2] or spec[3] == -1 else spec[3]
                waiting.setdefault(owner, []).append((index, obs))
            except RuleClarificationRequired as exc:
                questions.append(
                    {
                        "seed": spec[0],
                        "seat": spec[2],
                        "step": steps,
                        "error": str(exc),
                        "state": state.serialize(),
                    }
                )
                del active[index]
        for owner, items in waiting.items():
            observations = [obs for _, obs in items]
            details = {} if audit is not None and owner == -1 else None
            indices, logp, values, features, actions = infer(
                model if owner == -1 else opponents[owner],
                observations,
                device,
                generator if training and owner == -1 else None,
                details,
                temperature=sampling_temperature if training and owner == -1 else 1.0,
            )
            relevant = [
                j
                for j, obs in enumerate(observations)
                if details is not None and decision_audit.category(obs)
            ]
            parent_choices = {}
            if relevant:
                choices = infer(audit_parent or model, [observations[j] for j in relevant], device)[
                    0
                ]
                parent_choices = dict(zip(relevant, choices, strict=True))
            for j, (index, obs) in enumerate(items):
                state, spec, trajectories, steps, audit_trail = active[index]
                action = obs.legal_actions[indices[j]]
                if owner == -1 and decision_recorder is not None:
                    audit_trail.append(decision_recorder(obs, j, indices[j], spec, steps + 1))
                if j in parent_choices:
                    hu = next((a for a in obs.legal_actions if a.kind.value == "hu"), None)
                    cash = engine.step(state, hu).settlement.payments[obs.player] if hu else None
                    audit_trail.append(
                        decision_audit.record(
                            obs,
                            indices[j],
                            parent_choices[j],
                            details["probabilities"][j][: len(obs.legal_actions)],
                            spec=spec,
                            step=steps + 1,
                            immediate_hu_cash=cash,
                        )
                    )
                if training and owner == -1:
                    trajectories[obs.player].append(
                        (features[j], actions[j], indices[j], logp[j], values[j])
                    )
                try:
                    active[index][0] = engine.step(state, action)
                    active[index][3] += 1
                except RuleClarificationRequired as exc:
                    questions.append(
                        {
                            "seed": spec[0],
                            "seat": spec[2],
                            "error": str(exc),
                            "state": state.serialize(),
                        }
                    )
                    del active[index]
    return rows, results, questions


def ppo_update(
    model,
    anchor,
    optimizer,
    rows,
    rng,
    *,
    device,
    epochs=3,
    batch_size=256,
    value_coef=0.5,
    entropy_coef=0.01,
    anchor_coef=0.02,
    sampling_temperature=1.0,
):
    if not np.isfinite(sampling_temperature) or sampling_temperature <= 0:
        raise ValueError("sampling temperature must be finite and positive")
    if any(not np.isfinite(c) or c < 0 for c in (value_coef, entropy_coef, anchor_coef)):
        raise ValueError("PPO loss coefficients must be finite and nonnegative")
    if not rows:
        raise RuntimeError("no valid on-policy decisions")
    advantages = np.array([r[5] - r[4] for r in rows], np.float32)
    advantages = (advantages - advantages.mean()) / max(float(advantages.std()), 0.1)
    metrics = []
    model.train()
    stopped = False
    for _ in range(epochs):
        order = list(range(len(rows)))
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            selected = [rows[i] for i in idx]
            x, a, mask = pack([r[0] for r in selected], [r[1] for r in selected], device)
            logits, value, _ = model(x, a, mask)
            logits = logits / sampling_temperature
            with torch.no_grad():
                reference = (anchor(x, a, mask)[0] / sampling_temperature).log_softmax(-1)
            distribution = torch.distributions.Categorical(logits=logits)
            chosen = torch.tensor([r[2] for r in selected], device=device)
            logprob = distribution.log_prob(chosen)
            old = torch.tensor([r[3] for r in selected], device=device)
            adv = torch.from_numpy(advantages[idx]).to(device)
            logratio = logprob - old
            ratio = logratio.exp()
            approximate_kl = (ratio - 1 - logratio).mean()
            if float(approximate_kl.detach()) > 0.03:
                stopped = True
                break
            actor = -torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
            target = torch.tensor([r[5] for r in selected], device=device)
            critic = nn.functional.mse_loss(value, target)
            anchor_kl = (reference.exp() * (reference - logits.log_softmax(-1))).sum(-1).mean()
            entropy = distribution.entropy().mean()
            loss = actor + value_coef * critic - entropy_coef * entropy + anchor_coef * anchor_kl
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite cash PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            metrics.append(
                {
                    "loss": float(loss.detach()),
                    "critic": float(critic.detach()),
                    "kl": float(approximate_kl.detach()),
                    "actor": float(actor.detach()),
                    "entropy": float(entropy.detach()),
                    "anchor_kl": float(anchor_kl.detach()),
                }
            )
        if stopped:
            break
    model.eval()
    return {
        "updates": len(metrics),
        "kl_stopped": stopped,
        **{
            k: sum(m[k] for m in metrics) / max(1, len(metrics))
            for k in ("loss", "critic", "kl", "actor", "entropy", "anchor_kl")
        },
    }


def paired_summary(candidate, reference, seed, alpha=0.05):
    left = {(r["seed"], r["seat"], r["opponent"]): r for r in candidate}
    right = {(r["seed"], r["seat"], r["opponent"]): r for r in reference}
    if left.keys() != right.keys():
        raise RuntimeError("paired evaluation has unmatched or excluded games")
    summaries = {}
    for opponent in sorted({key[2] for key in left}):
        seeds = sorted({key[0] for key in left if key[2] == opponent})
        delta = np.array(
            [
                np.mean(
                    [
                        left[(s, seat, opponent)]["cash"] - right[(s, seat, opponent)]["cash"]
                        for seat in range(3)
                    ]
                )
                for s in seeds
            ]
        )
        rng = np.random.default_rng(seed + opponent)
        boot = np.mean(rng.choice(delta, (10000, len(delta)), replace=True), axis=1)
        summaries[str(opponent)] = {
            "seeds": len(seeds),
            "games": 3 * len(seeds),
            "mean_cash_delta": float(delta.mean()),
            "paired_seed_bootstrap_95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "paired_seed_bootstrap_interval": np.quantile(
                boot, [alpha / 2, 1 - alpha / 2]
            ).tolist(),
            "alpha": alpha,
            "candidate_mean_cash": float(
                np.mean([r["cash"] for k, r in left.items() if k[2] == opponent])
            ),
            "base_mean_cash": float(
                np.mean([r["cash"] for k, r in right.items() if k[2] == opponent])
            ),
            "changed_trajectory_games": sum(
                left[k].get("history_sha256") != right[k].get("history_sha256")
                for k in left
                if k[2] == opponent
            ),
        }
    return summaries


def aggregate_summary(candidate, reference, seed, alpha=0.05):
    """Cluster all seats AND opponents by seed; bootstrap is approximate."""
    paired_summary(candidate, reference, seed)  # Reject incomplete pairs.
    right = {(r["seed"], r["seat"], r["opponent"]): r["cash"] for r in reference}
    seeds = sorted({r["seed"] for r in candidate})
    values = np.array(
        [
            np.mean(
                [
                    r["cash"] - right[(s, r["seat"], r["opponent"])]
                    for r in candidate
                    if r["seed"] == s
                ]
            )
            for s in seeds
        ]
    )
    rng = np.random.default_rng(seed)
    samples = np.mean(rng.choice(values, (10000, len(values)), replace=True), axis=1)
    return {
        "independent_seeds": len(seeds),
        "mean_cash_delta": float(values.mean()),
        "paired_seed_bootstrap_interval": np.quantile(samples, [alpha / 2, 1 - alpha / 2]).tolist(),
        "alpha": alpha,
        "interval_method": "approximate seed-cluster bootstrap",
    }


def evaluation(model, base, opponents, start, count, device, slots, *, audit=False, alpha=0.05):
    specs = [
        (s, s % 3, seat, opponent)
        for opponent in range(len(opponents))
        for s in range(start, start + count)
        for seat in range(3)
    ]
    candidate_audit, reference_audit = ([] if audit else None), ([] if audit else None)
    _, candidate, questions = play(
        model,
        opponents,
        specs,
        device=device,
        slots=slots,
        audit=candidate_audit,
        audit_parent=base,
    )
    _, reference, other_questions = play(
        base, opponents, specs, device=device, slots=slots, audit=reference_audit, audit_parent=base
    )
    if questions or other_questions:
        return {"valid": False, "questions": questions + other_questions}
    result = {"valid": True, "opponents": paired_summary(candidate, reference, start, alpha)}
    result["aggregate"] = aggregate_summary(candidate, reference, start, alpha)
    if audit:
        result["candidate_decisions"] = decision_audit.summarize(candidate_audit, stochastic=False)
        result["parent_decisions"] = decision_audit.summarize(reference_audit, stochastic=False)
    return result


def training_specs(start, count, opponents, selfplay_fraction, seed):
    """Assign match types independently of dealer/seat, with balanced pool coverage."""
    selfplay = round(count * selfplay_fraction)
    owners = [-1] * selfplay + [i % opponents for i in range(count - selfplay)]
    random.Random(seed).shuffle(owners)
    return [(start + i, i % 3, (i // 3) % 3, owner) for i, owner in enumerate(owners)]


def atomic_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    with temporary.open("rb") as file:
        os.fsync(file.fileno())
    os.replace(temporary, path)


def source_hashes():
    return {
        str(p): digest(p)
        for package in ("engine", "model", "training")
        for p in sorted((Path("zimortal") / package).glob("*.py"))
    }


def commit_report(path, iteration, title=None):
    # Stage only this report; never commit unrelated user-staged changes.
    if subprocess.run(["git", "diff", "--cached", "--quiet"], check=False).returncode:
        return "skipped: existing staged changes"
    subprocess.run(["git", "add", "--", str(path)], check=True)
    if not subprocess.run(["git", "diff", "--cached", "--quiet"], check=False).returncode:
        return "unchanged report already committed"
    subprocess.run(
        ["git", "commit", "-m", title or f"train(rl): 记录现金PPO第{iteration}代训练与评估"],
        check=True,
    )
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required, refusing silent CPU fallback")
    root = Path(args.directory)
    root.mkdir(parents=True, exist_ok=True)
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    configuration = {k: v for k, v in vars(args).items() if k != "commit_reports"}
    configuration.update(
        sources=source_hashes(),
        base_sha256=digest(args.base),
        historical_sha256=digest(args.historical),
        initialize_sha256=digest(args.initialize) if args.initialize else None,
        optimizer_checkpoint_sha256=digest(args.optimizer_checkpoint)
        if args.optimizer_checkpoint
        else None,
        opponent_sha256={p: digest(p) for p in args.opponents},
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0) if args.device == "cuda" else None,
    )
    config_path = root / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError("frozen RL config, sources, environment or parents changed")
    atomic_json(config_path, configuration)
    model = load_model(args.initialize or args.base, args.device)
    if model.architecture not in ("resnet", "mlp"):
        raise ValueError("cash PPO requires a policy/value network")
    base = load_model(args.base, args.device)
    if (model.architecture, model.feature_version, model.width) != (
        base.architecture,
        base.feature_version,
        base.width,
    ):
        raise ValueError("warm-start model and cash anchor must have matching encoders")
    historical = load_model(args.historical, args.device)
    frozen_paths = [args.base, args.historical] + args.opponents
    if len(set(frozen_paths)) != len(frozen_paths):
        raise ValueError("opponent paths must be distinct")
    frozen_opponents = [base, historical] + [load_model(p, args.device) for p in args.opponents]
    metadata = torch.load(args.initialize or args.base, map_location="cpu", weights_only=True).get(
        "metadata", {}
    )
    value_reset = "payments/100" not in metadata.get("value_target", "")
    if value_reset:
        nn.init.zeros_(model.value[-1].weight)
        nn.init.zeros_(model.value[-1].bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if args.optimizer_checkpoint:
        previous = torch.load(
            args.optimizer_checkpoint, map_location=args.device, weights_only=False
        )
        state_key = "best_state" if args.optimizer_best else "model"
        if not all(torch.equal(v, previous[state_key][k]) for k, v in model.state_dict().items()):
            raise ValueError("continued optimizer must correspond exactly to initial weights")
        optimizer.load_state_dict(
            previous["best_optimizer" if args.optimizer_best else "optimizer"]
        )
        for group in optimizer.param_groups:
            group["lr"] = args.lr
    rng = random.Random(args.seed)
    start, best_score = 0, -float("inf")
    best_state, best_optimizer = (
        copy.deepcopy(model.state_dict()),
        copy.deepcopy(optimizer.state_dict()),
    )
    last = root / "last.pt"
    if last.exists():
        saved = torch.load(last, map_location=args.device, weights_only=False)
        if saved["configuration"] != configuration:
            raise ValueError("resume checkpoint config mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start, best_score = saved["iteration"], saved["best_score"]
        best_state, best_optimizer = saved["best_state"], saved["best_optimizer"]
        rng.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if args.device == "cuda":
            torch.cuda.set_rng_state_all([x.cpu() for x in saved["cuda_rng"]])
    for iteration in range(start, args.iterations):
        if (
            source_hashes() != configuration["sources"]
            or digest(args.base) != configuration["base_sha256"]
            or digest(args.historical) != configuration["historical_sha256"]
            or any(digest(p) != sha for p, sha in configuration["opponent_sha256"].items())
            or (args.initialize and digest(args.initialize) != configuration["initialize_sha256"])
            or (
                args.optimizer_checkpoint
                and digest(args.optimizer_checkpoint)
                != configuration["optimizer_checkpoint_sha256"]
            )
        ):
            raise ValueError("frozen RL producer or opponents changed")
        started = time.monotonic()
        atomic_json(
            root / "progress.json",
            {
                "status": "collecting",
                "iteration": iteration + 1,
                "iterations": args.iterations,
                "device": args.device,
            },
        )
        snapshots = [
            root / f"generation-{n:03d}.pt"
            for n in range(max(1, iteration - args.snapshot_pool + 1), iteration + 1)
        ]
        opponents = frozen_opponents + [load_model(p, args.device) for p in snapshots]
        opponent_names = frozen_paths + [str(p) for p in snapshots]
        specs = training_specs(
            args.train_start + iteration * args.games,
            args.games,
            len(opponents),
            args.selfplay_fraction,
            args.seed + iteration,
        )
        training_audit = [] if args.audit_decisions else None
        rows, games, questions = play(
            model,
            opponents,
            specs,
            device=args.device,
            slots=args.slots,
            seed=args.seed + iteration,
            training=True,
            audit=training_audit,
            audit_parent=base,
            sampling_temperature=getattr(args, "sampling_temperature", 1.0),
        )
        if questions:
            atomic_json(root / f"questions-{iteration + 1:03d}.json", questions)
        if len(questions) > max(2, args.games // 20):
            raise RuntimeError("too many undefined-rule games; inspect saved questions")
        metrics = ppo_update(
            model,
            base,
            optimizer,
            rows,
            rng,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            value_coef=getattr(args, "value_coef", 0.5),
            entropy_coef=getattr(args, "entropy_coef", 0.01),
            anchor_coef=getattr(args, "anchor_coef", 0.02),
            sampling_temperature=getattr(args, "sampling_temperature", 1.0),
        )
        path = root / f"generation-{iteration + 1:03d}.pt"
        save_model(
            model,
            path,
            value_target="undiscounted terminal payments/100",
            algorithm="cash PPO",
            generation=iteration + 1,
        )
        dev = evaluation(
            model,
            base,
            frozen_opponents,
            args.dev_start,
            args.dev_seeds,
            args.device,
            args.slots,
            audit=args.audit_decisions,
        )
        score = (
            np.mean([r["mean_cash_delta"] for r in dev["opponents"].values()])
            if dev["valid"]
            else -float("inf")
        )
        # Continue the learner; development selects a separate frozen snapshot.
        selected = bool(score > best_score)
        if selected:
            best_score = float(score)
            best_state, best_optimizer = (
                copy.deepcopy(model.state_dict()),
                copy.deepcopy(optimizer.state_dict()),
            )
            save_model(
                model,
                root / "best-dev.pt",
                value_target="undiscounted terminal payments/100",
                generation=iteration + 1,
                development_selection=True,
            )
        evidence = hashlib.sha256()
        for row in rows:
            evidence.update(row[0].tobytes())
            evidence.update(row[1].tobytes())
            evidence.update(np.asarray(row[2:], np.float64).tobytes())
        decision_hash = evidence.hexdigest()
        report = {
            "decision_sha256": decision_hash,
            "value_output_reset": value_reset,
            "generation": iteration + 1,
            "device": args.device,
            "gpu": configuration["gpu"],
            "completed_games": len(games),
            "opponent_pool": opponent_names,
            "opponent_sha256": {str(p): digest(p) for p in snapshots},
            "games_by_opponent": {
                str(owner): sum(r["opponent"] == owner for r in games)
                for owner in range(-1, len(opponents))
            },
            "training_wins": sum(r["winner"] == r["seat"] for r in games),
            "training_draws": sum(r["winner"] is None for r in games),
            "training_cash_per_game": sum(r["cash"] for r in games) / max(1, len(games)),
            "excluded_rule_games": len(questions),
            "decisions": len(rows),
            "nonzero_return_decisions": sum(r[5] != 0 for r in rows),
            "reward": "undiscounted terminal net cash / 100; no reward clipping",
            "selfplay_and_frozen_opponents": True,
            "replay_verified": True,
            "training_start": args.train_start + iteration * args.games,
            "metrics": metrics,
            "decision_audit": decision_audit.summarize(training_audit, stochastic=True)
            if training_audit is not None
            else None,
            "development": dev,
            "selected_on_development": selected,
            "cumulative_training": True,
            "formal_champion_changed": False,
            "checkpoint": str(path),
            "sha256": digest(path),
            "elapsed_seconds": time.monotonic() - started,
        }
        report_path = reports / f"generation-{iteration + 1:03d}.json"
        atomic_json(report_path, report)
        atomic_checkpoint(
            last,
            {
                "configuration": configuration,
                "decision_sha256": decision_hash,
                "iteration": iteration + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_score": best_score,
                "best_state": best_state,
                "best_optimizer": best_optimizer,
                "python_rng": rng.getstate(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if args.device == "cuda" else [],
            },
        )
        atomic_json(root / "progress.json", {"status": "iteration_complete", **report})
        print(json.dumps(report, ensure_ascii=False), flush=True)
        if args.commit_reports:
            commit_report(report_path, iteration + 1)
    atomic_json(
        root / "progress.json",
        {"status": "holdout_evaluation", "iterations": args.iterations, "device": args.device},
    )
    if not (root / "best-dev.pt").exists():
        raise RuntimeError("no valid development candidate for independent evaluation")
    model.load_state_dict(best_state)
    holdout = evaluation(
        model,
        base,
        frozen_opponents,
        args.holdout_start,
        args.holdout_seeds,
        args.device,
        args.slots,
        audit=args.audit_decisions,
        alpha=args.holdout_alpha,
    )
    atomic_json(
        reports / "holdout.json",
        {
            "evaluation": holdout,
            "checkpoint": str(root / "best-dev.pt"),
            "formal_champion_changed": False,
            "selected_by_development_only": True,
            "opponent_pool": frozen_paths,
        },
    )
    if args.commit_reports:
        commit_report(
            reports / "holdout.json", args.iterations, "train(rl): 记录现金PPO独立留出收益评估"
        )
    atomic_json(
        root / "progress.json",
        {
            "status": "complete",
            "iterations": args.iterations,
            "device": args.device,
            "best_dev_cash_delta": best_score,
            "holdout": holdout,
            "formal_champion_changed": False,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="checkpoints/huxi-resnet.pt")
    parser.add_argument("--initialize")
    parser.add_argument("--optimizer-checkpoint")
    parser.add_argument("--optimizer-best", action="store_true")
    parser.add_argument("--holdout-alpha", type=float, default=0.05)
    parser.add_argument("--historical", default="checkpoints/scale100k-resnet.pt")
    parser.add_argument("--opponents", nargs="*", default=[])
    parser.add_argument("--snapshot-pool", type=int, default=0)
    parser.add_argument("--selfplay-fraction", type=float, default=1 / 3)
    parser.add_argument("--audit-decisions", action="store_true")
    parser.add_argument("--directory", default="data/generated/online-rl-v1")
    parser.add_argument("--reports", default="docs/training/online-rl-v1")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--slots", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--anchor-coef", type=float, default=0.02)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=108)
    parser.add_argument("--train-start", type=int, default=4000000)
    parser.add_argument("--dev-start", type=int, default=5000000)
    parser.add_argument("--dev-seeds", type=int, default=30)
    parser.add_argument("--holdout-start", type=int, default=6000000)
    parser.add_argument("--holdout-seeds", type=int, default=100)
    parser.add_argument("--commit-reports", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.sampling_temperature) or args.sampling_temperature <= 0:
        parser.error("sampling temperature must be finite and positive")
    if any(
        not np.isfinite(c) or c < 0 for c in (args.value_coef, args.entropy_coef, args.anchor_coef)
    ):
        parser.error("PPO loss coefficients must be finite and nonnegative")
    if not 0 < args.holdout_alpha < 1:
        parser.error("holdout alpha must be in (0,1)")
    if not 0 <= args.selfplay_fraction <= 1 or args.snapshot_pool < 0:
        parser.error("selfplay fraction must be in [0,1] and snapshot pool nonnegative")
    if (
        min(
            args.games,
            args.slots,
            args.iterations,
            args.epochs,
            args.batch_size,
            args.dev_seeds,
            args.holdout_seeds,
        )
        < 1
    ):
        parser.error("counts must be positive")
    if (
        not args.train_start + args.games * args.iterations <= args.dev_start
        or not args.dev_start + args.dev_seeds <= args.holdout_start
    ):
        parser.error("train, development and holdout seeds must not overlap")
    run(args)


if __name__ == "__main__":
    main()
