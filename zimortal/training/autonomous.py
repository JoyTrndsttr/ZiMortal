"""Independent contingent-Q pilots followed by an iterative real-cash PPO league.

Pilot transitions are auxiliary noisy MC regression, never certified root
labels. Search/test roots, particles and PPO seeds are disjoint. No human
review roots enter training. All metrics/evidence stay in generated storage.
"""

import argparse
import gzip
import itertools
import json
import math
import os
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

import numpy as np
import torch

from zimortal.belief.sampling import SamplingFailure
from zimortal.engine import ActionType as A
from zimortal.engine import RuleClarificationRequired, RuleEngine
from zimortal.model.cashq import CashQNet
from zimortal.model.encoding import encode_action, encode_observation

from .active import atomic_json, deserialize_observation, serialize_observation, tags
from .branch_teacher import (
    action_data,
    compact_evidence,
    evaluate_two_step_branches,
    producer_hashes,
    visible_key,
)
from .cashq import batch, digest
from .league import league
from .planning import atomic_save, training_device
from .runtime import choose, choose_batch, load_model, predict, save_model

DISK_RESERVE = 2 * 1024**3


def atomic_gzip(path, value):
    """Compact auditable evidence, retaining the existing two-GiB reserve."""
    path = Path(path)
    if shutil.disk_usage(path.parent).free < DISK_RESERVE:
        raise OSError("insufficient disk reserve; stop before writing evidence")
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as file:
        with gzip.GzipFile(fileobj=file, mode="wb", mtime=0) as compressed:
            compressed.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def mine_root(seed, parent, historical, preferred, mining_model=None):
    """One prioritized visible root from a fully verified independent deal."""
    engine = RuleEngine()
    initial = engine.new_game(seed, seed % 3)
    state, rng, candidates = initial, random.Random(seed), []
    for step in range(1, 1001):
        if state.terminal:
            break
        actions = engine.legal_actions(state)
        obs = engine.observation(state, actions[0].player)
        if len(actions) > 1:
            kinds = {a.kind for a in actions}
            family = (
                "hu"
                if A.HU in kinds
                else "chi"
                if A.CHI in kinds
                else "peng"
                if A.PENG in kinds
                else "discard"
            )
            if family != "discard" or obs.remaining_tiles <= 6:
                logits, _ = predict(parent, obs)
                probabilities = logits.softmax(-1).cpu().numpy()
                uncertainty = 1 - float(probabilities.max())
                score = 20 * (family == preferred) + 4 * (family != "discard")
                score += uncertainty + 2 * (obs.remaining_tiles <= 6)
                score += 0.5 * any(a.bi for a in actions)
                candidates.append((score, obs, step, family))
        behavior = historical if (actions[0].player + seed) % 3 == 0 else (mining_model or parent)
        state = engine.step(state, choose(obs, rng, behavior))
        state.validate()
    else:
        raise RuntimeError("mining game exceeded transition limit")
    if engine.replay(initial, state.history).serialize() != state.serialize():
        raise RuntimeError("mining full replay mismatch")
    if not candidates:
        return None
    _, obs, step, family = max(candidates, key=lambda row: (row[0], -row[2]))
    return obs, {
        "seed": seed,
        "step": step,
        "family": family,
        "tags": tags(obs),
        "observation_sha256": visible_key(obs),
        "physical_replay_verified": True,
    }


def search_rows(raw):
    """Only visible a2 states and terminal cash; no hidden features or oracle max."""
    groups = {}
    for particle in raw:
        for branch in particle["branches"]:
            if branch["observation"] is None:
                continue
            key = branch["information_set"]
            group = groups.setdefault(key, [branch["observation"], defaultdict(list)])
            group[1][particle["particle"]].append(branch)
    rows = []
    for observation, samples in groups.values():
        obs = deserialize_observation(observation)
        ordered = list(samples.values())
        actions = [action_data(a) for a in obs.legal_actions]
        if any([b["second_action"] for b in row] != actions for row in ordered):
            raise ValueError("inconsistent public information-set actions")
        cash = np.asarray([[b["cash"] for b in row] for row in ordered], dtype=np.float64)
        n = len(cash)
        q = cash.mean(0) / 100
        # Singleton uncertainty is unknown, never zero. This is a loss weight,
        # not a certified standard error or confidence interval.
        se = cash.std(0, ddof=1) / math.sqrt(n) / 100 if n > 1 else np.full(len(actions), 3.0)
        rows.append(
            {
                "x": encode_observation(obs, "huxi"),
                "actions": np.stack([encode_action(a, obs.player) for a in obs.legal_actions]),
                "q": q.astype(np.float32),
                "se": se.astype(np.float32),
                "soft": np.full(len(actions), 1 / len(actions), np.float32),
                "support": n,
            }
        )
    # Equal total loss weight per root; singleton-heavy PASS cannot swamp
    # repeated static claim/discard information sets.
    total = sum(r["support"] for r in rows)
    for row in rows:
        row["root_weight"] = row["support"] / total
    return rows


def fit_contingent(parent_path, roots, output, *, device, epochs=12, initialize=None):
    """Auxiliary Q fitting on search-only roots. Evaluation never updates weights."""
    parent = load_model(parent_path)
    if parent.architecture != "resnet" or parent.auxiliary_version != "legacy":
        raise ValueError("contingent pilot requires legacy huxi ResNet parent")
    model = CashQNet(parent.width).to(device)
    model.parent.load_state_dict(parent.state_dict())
    optimizer = torch.optim.AdamW(model.q_heads.parameters(), lr=3e-4)
    if initialize:
        saved = torch.load(initialize, map_location=device, weights_only=True)
        if saved["parent_sha256"] != digest(parent_path):
            raise ValueError("contingent parent changed")
        if digest(saved["checkpoint"]) != saved["checkpoint_sha256"]:
            raise ValueError("optimizer no longer matches its contingent checkpoint")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
    rows = [r for root in roots for r in root]
    if not rows:
        raise ValueError("no visible second-decision examples")
    before = {k: v.clone() for k, v in model.parent.state_dict().items()}
    logs = []
    rng = random.Random(760)
    for epoch in range(epochs):
        order = list(rows)
        rng.shuffle(order)
        losses = []
        model.train()
        model.parent.eval()
        for start in range(0, len(order), 64):
            selected = order[start : start + 64]
            x, actions, mask, q, se, _ = batch(selected, device)
            ensemble, _ = model.forward_q(x, actions, mask)
            error = torch.nn.functional.smooth_l1_loss(
                ensemble, q[..., None].expand_as(ensemble), reduction="none", beta=0.2
            )
            weight = torch.tensor([r["root_weight"] for r in selected], device=device)
            per_root = ((error / (1 + se[..., None].square())) * mask[..., None]).sum((1, 2))
            per_root /= (mask.sum(1) * ensemble.shape[-1]).clamp_min(1)
            # Average over roots in the entire corpus, not over information sets.
            loss = (per_root * weight).sum() / len(roots)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite contingent loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.q_heads.parameters(), 2)
            optimizer.step()
            losses.append(float(loss.detach()))
        logs.append({"epoch": epoch + 1, "loss": sum(losses)})
    if any(not torch.equal(v, model.parent.state_dict()[k]) for k, v in before.items()):
        raise RuntimeError("frozen parent encoder changed")
    model.enabled.fill_(True)
    model.margin_cash.fill_(0)
    model.eval()
    save_model(
        model,
        output,
        experimental=True,
        purpose="search-only contingent cash-Q pilot",
        parent_sha256=digest(parent_path),
        no_certified_root_labels=True,
    )
    recovery = str(output) + ".optimizer.pt"
    atomic_save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "parent_sha256": digest(parent_path),
            "checkpoint": str(output),
            "checkpoint_sha256": digest(output),
        },
        recovery,
    )
    return {
        "roots": len(roots),
        "information_sets": len(rows),
        "epochs": epochs,
        "logs": logs,
        "frozen_encoder_verified": True,
        "device": device,
        "checkpoint": str(output),
        "checkpoint_sha256": digest(output),
        "optimizer": recovery,
    }


def independent_comparison(results, model, parent, *, alpha=0.05):
    """Select a2 only from visible test states; CI clusters by independent deal."""
    improvements, records = [], []
    for obs, result in results:
        observations = {}
        for particle in result["raw_particles"]:
            for branch in particle["branches"]:
                if branch["observation"] is not None:
                    observations.setdefault(branch["information_set"], branch["observation"])
        keys = list(observations)
        choices = {}
        for start in range(0, len(keys), 128):
            batch_keys = keys[start : start + 128]
            selected = choose_batch(
                model, [deserialize_observation(observations[k]) for k in batch_keys]
            )
            choices.update((k, action_data(a)) for k, a in zip(batch_keys, selected, strict=True))
        first = int(predict(parent, obs)[0].argmax())
        baseline, candidate, changed = [], [], 0
        for particle in result["raw_particles"]:
            branches = [b for b in particle["branches"] if b["first_index"] == first]
            selected = next(
                b for b in branches if b["second_action"] == choices.get(b["information_set"])
            )
            reference = next(b for b in branches if b["reference_selected"])
            baseline.append(reference["cash"])
            candidate.append(selected["cash"])
            changed += selected["second_action"] != reference["second_action"]
        delta = float(np.mean(np.asarray(candidate) - baseline))
        improvements.append(delta)
        records.append(
            {"mean_cash_delta": delta, "particles": len(baseline), "changed_particles": changed}
        )
    if len(improvements) < 2:
        raise ValueError("at least two independent test deals required")
    mean = float(np.mean(improvements))
    se = float(np.std(improvements, ddof=1) / math.sqrt(len(improvements)))
    z = NormalDist().inv_cdf(1 - alpha / 2)
    interval = [mean - z * se, mean + z * se]
    return {
        "mean_cash_delta": mean,
        "root_cluster_standard_error": se,
        "approximate_interval": interval,
        "independent_deals": len(improvements),
        "accepted": se > 0 and interval[0] > 0,
        "records": records,
        "formal_champion_changed": False,
        "limitation": "approximate normal interval across independent deals; pilot only",
    }


def generate(args, root, name, count, seed_start, parent, historical, frozen, mining_model=None):
    rows, results, records, failures = [], [], [], []
    directory = root / name
    directory.mkdir(exist_ok=True)
    index_path = directory / "index.json"
    hashes = json.loads(index_path.read_text()) if index_path.exists() else {}
    failures_path = directory / "failures.json"
    failures = json.loads(failures_path.read_text()) if failures_path.exists() else []
    failed_seeds = {r["seed"] for r in failures}
    for index in range(count * 6):
        if len(records) >= count:
            break
        if producer_hashes() != frozen["sources"] or any(
            digest(p) != h for p, h in frozen["weights"].items()
        ):
            raise ValueError("frozen pilot sources or weights changed")
        seed = seed_start + index
        if seed in failed_seeds:
            continue
        path = directory / f"{seed}.json.gz"
        try:
            if path.exists():
                if hashes.get(str(seed)) != digest(path):
                    raise ValueError("pilot evidence is missing its checksum or changed")
                with gzip.open(path, "rt") as file:
                    saved = json.load(file)
                obs = deserialize_observation(saved["observation"])
                result, meta = saved["result"], saved["meta"]
                evidence = result.pop("evidence")
                result["raw_particles"] = [
                    dict(
                        p,
                        branches=[
                            dict(b, observation=evidence["information_sets"][b["information_set"]])
                            for b in p["branches"]
                        ],
                    )
                    for p in evidence["particles"]
                ]
            else:
                mined = mine_root(
                    seed, parent, historical, ("chi", "peng", "hu")[len(records) % 3], mining_model
                )
                if mined is None:
                    continue
                obs, meta = mined
                began = time.perf_counter()
                atomic_json(
                    root / "progress.json",
                    {
                        "status": "pilot_generation",
                        "split": name,
                        "completed": len(records),
                        "target": count,
                        "active_seed": seed,
                        "active_step": meta["step"],
                        "active_family": meta["family"],
                        "phase": "sampling_and_two_step_rollouts",
                        "next_plan": "fit visible Q on search only; evaluate on independent test deals",
                    },
                )
                result = evaluate_two_step_branches(
                    obs,
                    model=parent,
                    rollouts=args.rollouts,
                    minimum=args.rollouts,
                    seed=seed * 104729,
                    max_attempts=args.max_attempts,
                    adaptive=False,
                )
                if not result["valid"]:
                    failures.append(
                        {"seed": seed, "status": result["status"], "error": result["error"]}
                    )
                    atomic_json(directory / "failures.json", failures)
                    failed_evidence = {k: v for k, v in result.items() if k != "raw_particles"}
                    failed_evidence["evidence"] = compact_evidence(result["raw_particles"])
                    atomic_gzip(
                        directory / f"failed-{seed}.json.gz",
                        {
                            "observation": serialize_observation(obs),
                            "meta": meta,
                            "result": failed_evidence,
                        },
                    )
                    continue
                meta["wall_seconds"] = time.perf_counter() - began
                compact = {k: v for k, v in result.items() if k not in ("raw_particles", "groups")}
                compact["evidence"] = compact_evidence(result["raw_particles"])
                atomic_gzip(
                    path,
                    {"observation": serialize_observation(obs), "meta": meta, "result": compact},
                )
                hashes[str(seed)] = digest(path)
                atomic_json(index_path, hashes)
            if (
                not result["valid"]
                or visible_key(obs) != meta["observation_sha256"]
                or meta["seed"] != seed
            ):
                raise ValueError("invalid saved pilot root provenance")
            rows.append(search_rows(result["raw_particles"]))
            results.append((obs, result))
            records.append(meta)
            atomic_json(
                root / "progress.json",
                {
                    "status": "pilot_generation",
                    "split": name,
                    "completed": len(records),
                    "target": count,
                    "failures": len(failures),
                    "last": meta,
                    "next_plan": "fit visible contingent Q on search; evaluate on separate deals/particles",
                },
            )
            print(
                json.dumps(
                    {
                        "split": name,
                        "completed": len(records),
                        "family": meta["family"],
                        "seed": seed,
                    }
                ),
                flush=True,
            )
        except (RuleClarificationRequired, SamplingFailure) as exc:
            failures.append(
                {
                    "seed": seed,
                    "error": str(exc),
                    "status": "undefined rule or infeasible posterior; whole game/root excluded",
                }
            )
            atomic_json(directory / "failures.json", failures)
    if len(records) < count:
        raise RuntimeError(
            f"insufficient valid independent pilot roots: {name}: {len(records)}/{count}"
        )
    return rows, results, records


def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    from zimortal.engine import game

    from .rollout_engine import cached_chi, cached_evaluate_hand

    game.evaluate_hand = cached_evaluate_hand
    game.enumerate_chi_with_required_bi = cached_chi
    device = training_device(args.device)
    root = Path(args.directory)
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "arguments": vars(args),
        "sources": producer_hashes(),
        "weights": {p: digest(p) for p in (args.parent, args.historical, args.initial_optimizer)},
    }
    path = root / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("autonomous configuration changed; use a new directory")
    atomic_json(path, config)
    parent, historical = load_model(args.parent), load_model(args.historical)
    corpus, rounds, optimizer = [], [], None
    mining_model = None
    adopted_teacher = None
    adopted_sha256 = None
    for i in range(args.rounds):
        count = args.roots * 2**i
        search_config = {
            **config,
            "weights": {
                **config["weights"],
                **({adopted_teacher: adopted_sha256} if adopted_teacher else {}),
            },
        }
        search, _, metadata = generate(
            args,
            root,
            f"search-{i + 1:02d}",
            count,
            args.search_start + i * 100000,
            parent,
            historical,
            search_config,
            mining_model,
        )
        corpus.extend(search)
        output = root / f"contingent-{i + 1:02d}.pt"
        trained = fit_contingent(
            args.parent, corpus, output, device=device, epochs=args.epochs, initialize=optimizer
        )
        optimizer = trained["optimizer"]
        atomic_json(
            root / "progress.json",
            {
                "status": "independent_contingent_evaluation",
                "round": i + 1,
                "trained": trained,
                "next_plan": "adopt only on fresh paired cash evidence; otherwise keep parent",
            },
        )
        _, test, testmeta = generate(
            args,
            root,
            f"test-{i + 1:02d}",
            args.test_roots,
            args.test_start + i * 100000,
            parent,
            historical,
            config,
        )
        if digest(output) != trained["checkpoint_sha256"]:
            raise ValueError("contingent checkpoint changed during independent test generation")
        compared = independent_comparison(
            test, load_model(output), parent, alpha=0.05 / args.rounds
        )
        if compared["accepted"]:
            mining_model = load_model(output)
            adopted_teacher = str(output)
            adopted_sha256 = trained["checkpoint_sha256"]
        report = {
            "round": i + 1,
            "adopted_mining_teacher": adopted_teacher,
            "search_records": metadata,
            "test_records": testmeta,
            "training": trained,
            "comparison": compared,
            "no_policy_distillation": True,
            "next_plan": "use accepted visible Q for next mining round; increase coverage"
            if compared["accepted"]
            else "retain mining behavior; increase search coverage; PPO uses retained cash baseline",
        }
        atomic_json(root / f"round-{i + 1:02d}.json", report)
        rounds.append(report)
        print(json.dumps({"round": i + 1, "contingent_comparison": compared}), flush=True)
    atomic_json(
        root / "pilot-summary.json",
        {
            "rounds": rounds,
            "certified_roots": 0,
            "no_manual_review_training": True,
            "no_formal_promotion": True,
            "adopted_mining_teacher": adopted_teacher,
            "next_plan": "real on-policy PPO; contingent pilots remain experimental and are not policy initialization",
        },
    )
    if args.ppo_generations:
        if shutil.disk_usage(root).free < DISK_RESERVE + 64 * 1024**2:
            raise OSError("insufficient reserve for PPO checkpoints")
        settings = argparse.Namespace(
            initial=args.parent,
            initial_optimizer=args.initial_optimizer,
            anchor="checkpoints/huxi-resnet.pt",
            historical=args.historical,
            opponents=[
                "data/generated/league-v4/stage-02/best-dev.pt",
                "data/generated/league-v3/stage-04/best-dev.pt",
                "checkpoints/planning-cuda-resnet.pt",
            ],
            directory=str(root / "league"),
            reports=str(root / "league-reports"),
            device=device,
            stages=args.rounds,
            generations=args.ppo_generations,
            games=args.ppo_games,
            later_games=args.ppo_games * 2,
            snapshot_pool=4,
            slots=64,
            lr=3e-5,
            selfplay_fraction=0.6,
            value_coef=0.1,
            entropy_coef=0.02,
            anchor_coef=0.01,
            sampling_temperature=1.5,
            seed=args.seed + 1000,
            train_start=args.ppo_start,
            dev_start=args.ppo_start + 10000000,
            screen_start=args.ppo_start + 20000000,
            final_start=args.ppo_start + 30000000,
            seed_stride=1000000,
            dev_seeds=80,
            screen_seeds=400,
            final_seeds=1000,
            regression_tolerance=3.0,
            commit_reports=False,
        )
        atomic_json(
            root / "progress.json",
            {
                "status": "ppo_league",
                "completed_pilot_rounds": len(rounds),
                "next_plan": "stage cash gates choose next model automatically; no per-generation commits",
                "league_progress": str(root / "league" / "league-progress.json"),
            },
        )
        league(settings)
    atomic_json(
        root / "progress.json",
        {
            "status": "complete",
            "pilot_rounds": len(rounds),
            "next_plan": "review independent cash results and failed roots before higher-quality root scaling",
            "formal_champion_changed": False,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="data/generated/autonomous-v1")
    parser.add_argument("--parent", default="data/generated/league-v2/stage-02/best-dev.pt")
    parser.add_argument("--initial-optimizer", default="data/generated/league-v2/stage-02/last.pt")
    parser.add_argument("--historical", default="checkpoints/scale100k-resnet.pt")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--roots", type=int, default=8)
    parser.add_argument("--test-roots", type=int, default=8)
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--max-attempts", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=812)
    parser.add_argument("--search-start", type=int, default=320000000)
    parser.add_argument("--test-start", type=int, default=350000000)
    parser.add_argument("--ppo-start", type=int, default=400000000)
    parser.add_argument("--ppo-generations", type=int, default=4)
    parser.add_argument("--ppo-games", type=int, default=2048)
    args = parser.parse_args()
    if (
        min(args.rounds, args.roots, args.test_roots, args.rollouts, args.epochs, args.max_attempts)
        < 2
    ):
        parser.error("pilot counts must be at least two")
    if (
        args.ppo_generations < 0
        or args.ppo_games < 1
        or args.rounds > 6
        or args.ppo_generations * args.ppo_games * 2 >= 1000000
        or args.roots * 2 ** (args.rounds - 1) >= 100000 // 6
    ):
        parser.error("invalid or overlapping round size")
    ranges = [
        (start, start + (args.rounds - 1) * 100000 + count * 6)
        for start, count in (
            (args.search_start, args.roots * 2 ** (args.rounds - 1)),
            (args.test_start, args.test_roots),
        )
    ]
    ranges += [(args.ppo_start, args.ppo_start + 40000000)]
    ranges.sort()
    if any(a[1] > b[0] for a, b in itertools.pairwise(ranges)):
        parser.error("search/test/PPO seeds overlap")
    try:
        run(args)
    except Exception as exc:
        root = Path(args.directory)
        root.mkdir(parents=True, exist_ok=True)
        atomic_json(
            root / "progress.json",
            {"status": "failed", "error": str(exc), "type": type(exc).__name__},
        )
        raise


if __name__ == "__main__":
    main()
