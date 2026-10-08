"""Simulator-only cash-Q iteration with diverse roots and held-out promotion."""

import argparse
import hashlib
import json
import random
import resource
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from zimortal.belief.sampling import SamplingFailure
from zimortal.engine import RuleEngine
from zimortal.model.cashq import CashQNet
from zimortal.model.encoding import encode_action, encode_observation

from .boundary_review import paired_interval
from .planning import atomic_save, training_device
from .rollout import teacher
from .runtime import choose, load_model, predict, save_model, tournament


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def root_for_game(seed, preferred, models):
    engine = RuleEngine()
    state = engine.new_game(seed, dealer=seed % 3)
    rng = random.Random(seed)
    stages = {"early": [], "middle": [], "late": []}
    behavior = seed % 4
    for step in range(1, 1001):
        if state.terminal:
            break
        acts = engine.legal_actions(state)
        obs = engine.observation(state, acts[0].player)
        if len(acts) > 1:
            stage = "early" if step <= 15 else "middle" if step <= 40 else "late"
            stages[stage].append((obs, step, stage))
        action = choose(
            obs, rng, models[behavior % len(models)], policy="teacher" if behavior == 3 else "model"
        )
        state = engine.step(state, action)
        state.validate()
    else:
        raise RuntimeError("root game did not terminate")
    if (
        engine.replay(engine.new_game(seed, seed % 3), state.history).serialize()
        != state.serialize()
    ):
        raise RuntimeError("root history replay mismatch")
    candidates = []
    # One successful root per independent deal. Try requested stage, then
    # fallback stages; report actual distribution and all failed attempts.
    for stage in [preferred] + [s for s in stages if s != preferred]:
        if stages[stage]:
            candidates.append(rng.choice(stages[stage]))
    return candidates, behavior


def build(args):
    torch.set_num_threads(2)
    root = Path(args.data)
    root.mkdir(parents=True, exist_ok=True)
    models = [
        load_model(p)
        for p in (args.parent, "checkpoints/scale100k-resnet.pt", "checkpoints/boundary-resnet.pt")
    ]
    config = {
        "roots": args.roots,
        "valid_roots": args.valid_roots,
        "rollouts": args.rollouts,
        "parent_sha256": digest(args.parent),
        "max_attempts": args.max_attempts,
        "behavior_sha256": [
            digest(p)
            for p in (
                args.parent,
                "checkpoints/scale100k-resnet.pt",
                "checkpoints/boundary-resnet.pt",
            )
        ],
    }
    config_path = root / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("preserve dataset configuration")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    records = {"train": [], "valid": []}
    known = set()
    failures = Counter()
    for split, first, count in (("train", 38000, args.roots), ("valid", 39800, args.valid_roots)):
        for index in range(count * 4):
            if len(records[split]) >= count:
                break
            seed = first + index
            path = root / f"{split}-{seed}.npz"
            meta_path = path.with_suffix(".json")
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                if meta["sha256"] != digest(path):
                    raise ValueError("root shard changed")
                if meta["input_hash"] in known:
                    continue
                records[split].append(meta)
                known.add(meta["input_hash"])
                continue
            candidates, behavior = root_for_game(
                seed, ("early", "middle", "late")[index % 3], models
            )
            for obs, step, stage in candidates:
                x = encode_observation(obs, "huxi")
                a = np.stack([encode_action(t, obs.player) for t in obs.legal_actions])
                key = hashlib.sha256(x.tobytes() + a.tobytes()).hexdigest()
                if key in known:
                    continue
                reference = int(predict(models[0], obs)[0].argmax())
                try:
                    result = teacher(
                        obs,
                        rollouts=args.rollouts,
                        seed=seed * 1000 + step,
                        model=models[0],
                        max_attempts=args.max_attempts,
                        batched=True,
                        reference_index=reference,
                    )
                except SamplingFailure:
                    failures[stage] += 1
                    continue
                np.savez_compressed(
                    path,
                    x=x,
                    actions=a,
                    q=np.array(result.q_cash, np.float32) / 100,
                    se=np.array(result.standard_errors, np.float32) / 100,
                    paired_se=np.array(result.paired_standard_errors, np.float32) / 100,
                    soft=np.array(result.policy, np.float32),
                    reference=reference,
                )
                meta = {
                    "seed": seed,
                    "dealer": seed % 3,
                    "step": step,
                    "player": obs.player,
                    "stage": stage,
                    "behavior": behavior,
                    "reference": reference,
                    "actions": len(a),
                    "rollouts_per_action": args.rollouts,
                    "sampling_attempts": result.sampling_attempts,
                    "file": path.name,
                    "sha256": digest(path),
                    "input_hash": key,
                }
                meta_path.write_text(json.dumps(meta, indent=2) + "\n")
                records[split].append(meta)
                known.add(key)
                print(
                    json.dumps(
                        {
                            "split": split,
                            "roots": len(records[split]),
                            "seed": seed,
                            "stage": stage,
                            "actions": len(a),
                        }
                    ),
                    flush=True,
                )
                break
        if len(records[split]) != count:
            raise RuntimeError(
                f"insufficient history-consistent {split} roots: {len(records[split])}"
            )
    manifest = {
        "config": config,
        "records": records,
        "sampling_failures_by_stage": dict(failures),
        "stages": {s: dict(Counter(r["stage"] for r in rows)) for s, rows in records.items()},
        "terminal_rollouts": {
            s: sum(r["actions"] * r["rollouts_per_action"] for r in rows)
            for s, rows in records.items()
        },
        "semantics": "net cash/100; uniform history-consistent particles; frozen huxi continuation",
        "source": {
            str(p): digest(p)
            for p in (
                Path(__file__),
                Path("zimortal/training/rollout.py"),
                Path("zimortal/training/runtime.py"),
            )
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def datasets(directory):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    splits = []
    for split in ("train", "valid"):
        rows = []
        for meta in manifest["records"][split]:
            path = root / meta["file"]
            if digest(path) != meta["sha256"]:
                raise ValueError("dataset changed")
            with np.load(path, allow_pickle=False) as data:
                rows.append({k: data[k] for k in data.files})
        splits.append(rows)
    return *splits, manifest


def audit(args):
    torch.set_num_threads(2)
    train, valid, manifest = datasets(args.data)
    keys = [{r["input_hash"] for r in manifest["records"][s]} for s in ("train", "valid")]
    if keys[0] & keys[1]:
        raise ValueError("cross-split duplicate")
    models = [
        load_model(p)
        for p in (args.parent, "checkpoints/scale100k-resnet.pt", "checkpoints/boundary-resnet.pt")
    ]
    checked = []
    label_checks = []
    for split, first, rows in (("train", 38000, train), ("valid", 39800, valid)):
        records = manifest["records"][split]
        if len({r["seed"] for r in records}) != len(rows):
            raise ValueError("multiple roots from one deal")
        for data in rows:
            n = len(data["actions"])
            if any(not np.isfinite(data[k]).all() for k in data):
                raise ValueError("nonfinite training label")
            if any(data[k].shape != (n,) for k in ("q", "se", "paired_se", "soft")):
                raise ValueError("invalid candidate label shape")
            if not np.isclose(data["soft"].sum(), 1) or np.any(data["se"] < 0):
                raise ValueError("invalid target probabilities or errors")
            if data["paired_se"][int(data["reference"])] != 0:
                raise ValueError("reference paired error must be zero")
        for index in range(0, len(rows), max(1, len(rows) // 8)):
            meta, data = records[index], rows[index]
            candidates, _ = root_for_game(
                meta["seed"], ("early", "middle", "late")[(meta["seed"] - first) % 3], models
            )
            obs = next(o for o, step, _stage in candidates if step == meta["step"])
            np.testing.assert_array_equal(encode_observation(obs, "huxi"), data["x"])
            np.testing.assert_array_equal(
                np.stack([encode_action(a, obs.player) for a in obs.legal_actions]), data["actions"]
            )
            checked.append({"split": split, "seed": meta["seed"], "step": meta["step"]})
            if index == 0:
                replayed = teacher(
                    obs,
                    rollouts=meta["rollouts_per_action"],
                    seed=meta["seed"] * 1000 + meta["step"],
                    model=models[0],
                    max_attempts=manifest["config"]["max_attempts"],
                    batched=True,
                    reference_index=meta["reference"],
                )
                for field, values in (
                    ("q", replayed.q_cash),
                    ("se", replayed.standard_errors),
                    ("paired_se", replayed.paired_standard_errors),
                ):
                    np.testing.assert_array_equal(np.array(values, np.float32) / 100, data[field])
                label_checks.append({"split": split, "seed": meta["seed"], "step": meta["step"]})
    Path(args.report).write_text(
        json.dumps(
            {
                "unique_roots": {"train": len(train), "valid": len(valid)},
                "reconstructed": checked,
                "cash_labels_recomputed_exactly": label_checks,
                "input_is_observation_only": True,
                "cross_split_duplicates": 0,
                "root_games_replay_and_conservation_verified": True,
            },
            indent=2,
        )
        + "\n"
    )


def batch(rows, device):
    size = max(len(r["actions"]) for r in rows)
    x = torch.from_numpy(np.stack([r["x"] for r in rows])).to(device)
    a = torch.zeros(len(rows), size, rows[0]["actions"].shape[-1], device=device)
    mask = torch.zeros(len(rows), size, dtype=torch.bool, device=device)
    q, se, soft = [torch.zeros(len(rows), size, device=device) for _ in range(3)]
    for i, r in enumerate(rows):
        n = len(r["actions"])
        a[i, :n] = torch.from_numpy(r["actions"]).to(device)
        mask[i, :n] = True
        for target, name in ((q, "q"), (se, "se"), (soft, "soft")):
            target[i, :n] = torch.from_numpy(r[name]).to(device)
    return x, a, mask, q, se, soft


def diagnostics(model, rows):
    device = next(model.parameters()).device
    error = total = regret = reference_regret = agreements = changed = 0
    with torch.inference_mode():
        for start in range(0, len(rows), 64):
            r = rows[start : start + 64]
            x, a, mask, q, _, _ = batch(r, device)
            ensemble, _ = model.forward_q(x, a, mask)
            mean = ensemble.mean(-1).masked_fill(~mask, -1e9)
            choice = mean.argmax(-1)
            reference = torch.tensor([int(t["reference"]) for t in r], device=device)
            row = torch.arange(len(r), device=device)
            best = q.masked_fill(~mask, -1e9).max(-1).values
            error += float((mean[mask] - q[mask]).square().sum())
            total += int(mask.sum())
            regret += float((best - q[row, choice]).sum()) * 100
            reference_regret += float((best - q[row, reference]).sum()) * 100
            agreements += int((choice == q.masked_fill(~mask, -1e9).argmax(-1)).sum())
            changed += int((choice != reference).sum())
    return {
        "action_cash_rmse": (error / total) ** 0.5 * 100,
        "mc_regret": regret / len(rows),
        "baseline_mc_regret": reference_regret / len(rows),
        "teacher_top1": agreements / len(rows),
        "changed_from_parent": changed / len(rows),
        "roots": len(rows),
    }


def train(args):
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(801)
    rng = random.Random(801)
    train_rows, valid, manifest = datasets(args.data)
    device = training_device(args.device)
    selection = getattr(args, "selection", "policy_regret")
    parent = load_model(args.parent)
    if (
        parent.architecture != "resnet"
        or parent.feature_version != "huxi"
        or parent.auxiliary_version != "legacy"
    ):
        raise ValueError("cash-Q requires the frozen legacy huxi ResNet baseline")
    model = CashQNet(parent.width).to(device)
    model.parent.load_state_dict(parent.state_dict())
    optimizer = torch.optim.AdamW(model.q_heads.parameters(), lr=3e-4, weight_decay=0.01)
    logs, best, first = [], float("inf"), 0
    recovery = args.output + ".recovery.pt"
    if args.continue_training:
        saved = torch.load(recovery, map_location=device, weights_only=True)
        if saved.get("selection", "cash_rmse") != selection:
            raise ValueError("resume checkpoint selection changed")
        if saved["manifest_sha256"] != digest(Path(args.data) / "manifest.json") or saved[
            "parent_sha256"
        ] != digest(args.parent):
            raise ValueError("resume inputs changed")
        if saved["best_model_sha256"] != digest(args.output):
            raise ValueError("best checkpoint changed")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.setstate(saved["rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device == "cuda":
            torch.cuda.set_rng_state_all([v.cpu() for v in saved["cuda_rng"]])
        logs, best, first = saved["logs"], saved["best"], saved["epoch"]
    for epoch in range(first, args.epochs):
        order = train_rows.copy()
        rng.shuffle(order)
        loss_values = []
        model.train()
        for start in range(0, len(order), 32):
            x, a, mask, q, se, soft = batch(order[start : start + 32], device)
            ensemble, _ = model.forward_q(x, a, mask)
            weights = (torch.rand(len(x), 1, 3, device=device) > 0.2).float()
            losses = nn.functional.smooth_l1_loss(
                ensemble, q[:, :, None].expand_as(ensemble), reduction="none", beta=0.1
            )
            # Give each root equal weight regardless of candidate count.
            numerator = (losses * weights * mask[:, :, None] / (1 + se[:, :, None].square())).sum(
                (1, 2)
            )
            denominator = (weights * mask[:, :, None]).sum((1, 2)).clamp_min(1)
            loss = (numerator / denominator).mean()
            # Common particles make action-minus-parent targets less noisy.
            current = order[start : start + 32]
            reference = torch.tensor([int(r["reference"]) for r in current], device=device)
            row = torch.arange(len(current), device=device)
            delta = ensemble - ensemble[row, reference][:, None]
            truth_delta = q - q[row, reference][:, None]
            paired = torch.zeros_like(q)
            for i, r in enumerate(current):
                paired[i, : len(r["paired_se"])] = torch.from_numpy(r["paired_se"]).to(device)
            pair_weight = mask / (1 + (paired / 0.1).square())
            pair_weight[row, reference] = 0
            pair_loss = nn.functional.smooth_l1_loss(
                delta, truth_delta[:, :, None].expand_as(delta), reduction="none", beta=0.05
            )
            pair_numerator = (pair_loss * pair_weight[:, :, None] * weights).sum((1, 2))
            pair_denominator = (pair_weight[:, :, None] * weights).sum((1, 2)).clamp_min(1)
            loss += 0.5 * (pair_numerator / pair_denominator).mean()
            # MC action ranking supplements regression; rewards are never clipped.
            logits = ensemble.mean(-1).masked_fill(~mask, -1e9) * 10
            loss += 0.03 * -(soft * logits.log_softmax(-1)).sum(-1).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite cash-Q loss")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.q_heads.parameters(), 2)
            optimizer.step()
            loss_values.append(float(loss.detach()))
        model.eval()
        validation = diagnostics(model, valid)
        score = (
            validation["mc_regret"] + 0.01 * validation["action_cash_rmse"]
            if selection == "policy_regret"
            else validation["action_cash_rmse"] + 0.1 * validation["mc_regret"]
        )
        if score < best:
            best = score
            save_model(
                model,
                args.output,
                value_target="candidate net cash/100; frozen huxi continuation",
                selection=selection,
                epoch=epoch + 1,
            )
        log = {
            "epoch": epoch + 1,
            "loss": sum(loss_values) / len(loss_values),
            "validation": validation,
        }
        logs.append(log)
        atomic_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best": best,
                "logs": logs,
                "rng": rng.getstate(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
                "manifest_sha256": digest(Path(args.data) / "manifest.json"),
                "parent_sha256": digest(args.parent),
                "best_model_sha256": digest(args.output),
                "selection": selection,
            },
            recovery,
        )
        print(json.dumps(log), flush=True)
    report = {
        "epochs": logs,
        "validation": diagnostics(load_model(args.output).to(device), valid),
        "manifest": manifest,
        "checkpoint": args.output,
        "sha256": digest(args.output),
        "device": device,
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "peak_tensor_vram_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
        "seconds": time.monotonic() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "parent_frozen": True,
        "human_records": 0,
        "selection": selection,
    }
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")


def evaluate(args):
    torch.set_num_threads(2)
    manifest = json.loads((Path(args.data) / "manifest.json").read_text())
    used = {r["seed"] for rows in manifest["records"].values() for r in rows}
    development_seeds = set(range(args.dev_start, args.dev_start + 30))
    holdout_seeds = set(range(args.eval_start, args.eval_start + args.eval_seeds))
    if development_seeds & holdout_seeds or (development_seeds | holdout_seeds) & used:
        raise ValueError("evaluation seeds overlap development or dataset roots")
    parent = load_model(args.parent)
    challenger = load_model(args.output)
    pool = {
        "frozen": ("model", parent),
        "teacher": ("teacher", None),
        "historical": ("model", load_model("checkpoints/scale100k-resnet.pt")),
        "random": ("random", None),
    }
    # Gate tuning uses development seeds exclusively. Holdout is used once.
    development = []
    for margin in (5, 10, 20):
        challenger.enabled.fill_(True)
        challenger.margin_cash.fill_(margin)
        score = 0
        for opponent in ("frozen", "teacher"):
            policy, model = pool[opponent]
            result = tournament(
                challenger,
                list(range(args.dev_start, args.dev_start + 30)),
                opponent=policy,
                opponent_model=model,
            )
            score += result["mean_payoff"]
        development.append({"margin": margin, "dev_payoff_sum": score})
        print(json.dumps(development[-1]), flush=True)
    selected = max(development, key=lambda x: x["dev_payoff_sum"])["margin"]
    challenger.margin_cash.fill_(selected)
    report = {
        "development": development,
        "development_seeds": [args.dev_start, args.dev_start + 29],
        "holdout_seeds": [args.eval_start, args.eval_start + args.eval_seeds - 1],
        "selected_margin_cash": selected,
        "opponents": {},
    }
    for name, (policy, model) in pool.items():
        seeds = list(range(args.eval_start, args.eval_start + args.eval_seeds))
        before = tournament(
            parent, seeds, opponent=policy, opponent_model=model, include_games=True
        )
        after = tournament(
            challenger, seeds, opponent=policy, opponent_model=model, include_games=True
        )
        paired = paired_interval(before["game_results"], after["game_results"])
        report["opponents"][name] = {"parent": before, "challenger": after, "paired": paired}
        print(
            json.dumps(
                {
                    "opponent": name,
                    "baseline": before["mean_payoff"],
                    "challenger": after["mean_payoff"],
                    "paired": paired,
                }
            ),
            flush=True,
        )
    # Each required opponent must clear a positive lower confidence bound.
    promote = all(
        report["opponents"][s]["paired"]["paired_bootstrap_95_percent"][0] > 0
        for s in ("frozen", "teacher", "historical")
    )
    report["promoted"] = promote
    report["promotion_rule"] = (
        "positive paired lower 95% bound vs frozen, teacher and historical opponents"
    )
    gated_path = str(Path(args.output).with_suffix(".gated.pt"))
    save_model(
        challenger,
        gated_path,
        value_target="candidate net cash/100; frozen huxi continuation",
        experimental=not promote,
        gate_margin_cash=selected,
    )
    report["checkpoint"] = gated_path
    report["ungated_training_checkpoint"] = args.output
    report["champion"] = gated_path if promote else args.parent
    champion = {
        "checkpoint": report["champion"],
        "sha256": digest(report["champion"]),
        "promoted": promote,
        "evidence": args.report,
    }
    Path("checkpoints/champion.json").write_text(json.dumps(champion, indent=2) + "\n")
    report["sha256"] = digest(gated_path)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["build", "audit", "train", "evaluate"])
    p.add_argument("--data", default="data/generated/cashq-v1")
    p.add_argument("--parent", default="checkpoints/huxi-resnet.pt")
    p.add_argument("--output", default="checkpoints/cashq-resnet.pt")
    p.add_argument("--report", default="docs/training/cashq-training.json")
    p.add_argument("--roots", type=int, default=256)
    p.add_argument("--valid-roots", type=int, default=64)
    p.add_argument("--rollouts", type=int, default=16)
    p.add_argument("--max-attempts", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--eval-seeds", type=int, default=100)
    p.add_argument("--dev-start", type=int, default=41000)
    p.add_argument("--eval-start", type=int, default=43000)
    p.add_argument("--selection", choices=["policy_regret", "cash_rmse"], default="policy_regret")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--continue-training", action="store_true")
    args = p.parse_args()
    if min(args.roots, args.valid_roots, args.epochs, args.eval_seeds, args.max_attempts) < 1:
        p.error("counts must be positive")
    if args.rollouts < 2:
        p.error("at least two rollouts are required")
    {"build": build, "audit": audit, "train": train, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
