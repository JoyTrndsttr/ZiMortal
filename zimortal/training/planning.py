"""Pilot rollout-policy, cash-EV, exact-distance and deck-belief training."""

import argparse
import hashlib
import json
import math
import os
import random
import resource
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from zimortal.analysis.completion import SearchLimit, analyze
from zimortal.belief.sampling import SamplingFailure
from zimortal.engine import RuleEngine
from zimortal.model.encoding import encode_action, encode_observation
from zimortal.model.network import PolicyValueNet

from .boundary import fixture
from .rollout import teacher
from .runtime import choose, load_model, save_model, tournament


def pack(obs, deck, soft=None, value=0.0, distance=-1):
    return (
        encode_observation(obs, "huxi"),
        np.stack([encode_action(a, obs.player) for a in obs.legal_actions]),
        soft,
        value,
        np.array([Counter(deck)[t] for t in range(20)], np.int64) if deck is not None else None,
        distance,
    )


def minibatch(rows, device="cpu"):
    x = torch.tensor(np.stack([r[0] for r in rows]))
    n = max(len(r[1]) for r in rows)
    actions = torch.zeros(len(rows), n, rows[0][1].shape[1])
    mask = torch.zeros(len(rows), n, dtype=torch.bool)
    targets = torch.zeros(len(rows), n)
    for i, r in enumerate(rows):
        size = len(r[1])
        actions[i, :size] = torch.tensor(r[1])
        mask[i, :size] = True
        if r[2] is not None:
            targets[i, :size] = torch.tensor(r[2])
    return tuple(t.to(device) for t in (x, actions, mask, targets))


def save_rows(path, rows):
    offsets = np.cumsum([0] + [len(r[1]) for r in rows])
    np.savez_compressed(
        path,
        x=np.stack([r[0] for r in rows]),
        actions=np.concatenate([r[1] for r in rows]),
        offsets=offsets,
        soft=np.concatenate([r[2] if r[2] is not None else np.zeros(len(r[1])) for r in rows]),
        rollout=np.array([r[2] is not None for r in rows]),
        value=np.array([r[3] for r in rows], np.float32),
        belief=np.stack([r[4] if r[4] is not None else np.full(20, -1) for r in rows]),
        distance=np.array([r[5] for r in rows]),
    )


def read_rows(path, max_bytes=1024**3):
    # Inspect uncompressed size before allocating; compressed file size is misleading.
    with zipfile.ZipFile(path) as archive:
        if sum(member.file_size for member in archive.infolist()) > max_bytes:
            raise ValueError("NPZ uncompressed data exceeds loader memory budget")
    with np.load(path, allow_pickle=False) as z:
        # NpzFile reloads an entire array on each lookup. Cache once so row
        # views share one backing array instead of retaining a copy per row.
        arrays = {
            name: z[name]
            for name in (
                "x",
                "offsets",
                "actions",
                "soft",
                "rollout",
                "value",
                "belief",
                "distance",
            )
        }
        rows = []
        for i, x in enumerate(arrays["x"]):
            lo, hi = arrays["offsets"][i : i + 2]
            rows.append(
                (
                    x,
                    arrays["actions"][lo:hi],
                    arrays["soft"][lo:hi] if arrays["rollout"][i] else None,
                    float(arrays["value"][i]),
                    arrays["belief"][i] if arrays["belief"][i, 0] >= 0 else None,
                    int(arrays["distance"][i]),
                )
            )
    return rows


def collect(start, games, model, rollout_states, rollouts):
    engine = RuleEngine()
    rows = []
    search = []
    seen = set()
    waiting = 0
    solved = 0
    limited = 0
    for seed in range(start, start + games):
        state = engine.new_game(seed, dealer=seed % 3)
        rng = random.Random(seed)
        for step in range(1, 1001):
            if state.terminal:
                break
            acts = engine.legal_actions(state)
            obs = engine.observation(state, acts[0].player)
            distance = -1
            if obs.phase == "draw" and waiting < 64:
                waiting += 1
                try:
                    answer = analyze(obs, max_nodes=500)
                    distance = answer.distance if answer.distance is not None else 15
                    solved += 1
                except SearchLimit:
                    limited += 1
            result = None
            # Search labels are fresh training roots, never the held-out human case.
            if (
                len(search) < rollout_states
                and seed - start >= len(search) * games // max(1, rollout_states)
                and 2 <= step <= 12
                and len(acts) > 1
            ):
                try:
                    result = teacher(
                        obs,
                        rollouts=rollouts,
                        seed=seed * 100 + step,
                        model=model,
                        max_attempts=2000,
                    )
                    search.append(
                        {
                            "seed": seed,
                            "step": step,
                            "player": obs.player,
                            "q_cash": result.q_cash,
                            "standard_errors": result.standard_errors,
                            "policy": result.policy,
                            "rollouts_per_action": rollouts,
                            "sampling_attempts": result.sampling_attempts,
                        }
                    )
                    print(
                        json.dumps({"search_roots": len(search), "seed": seed, "step": step}),
                        flush=True,
                    )
                except SamplingFailure:
                    pass
            r = pack(
                obs,
                state.deck,
                result.policy if result else None,
                sum(p * q for p, q in zip(result.policy, result.q_cash, strict=True)) / 100
                if result
                else 0,
                distance,
            )
            key = hashlib.sha256(r[0].tobytes() + r[1].tobytes()).digest()
            if key not in seen:
                rows.append(r)
                seen.add(key)
            state = engine.step(state, choose(obs, rng, model))
            state.validate()
        else:
            raise RuntimeError("data game did not terminate")
    return rows, {
        "games": games,
        "states": len(rows),
        "rollout_roots": search,
        "waiting_solver_attempts": waiting,
        "exact_waiting_labels": solved,
        "solver_limit_skipped": limited,
    }


def build(args):
    torch.set_num_threads(2)
    root = Path(args.data)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        raise ValueError("preserve existing dataset; use new directory")
    parent = load_model(args.resume)
    training, tm = collect(33000, args.games, parent, args.roots, args.rollouts)
    valid, vm = collect(
        34500, max(10, args.games // 5), parent, max(4, args.roots // 4), args.rollouts
    )
    # Replace corruption labels with exact intake solutions on an auxiliary set.
    for split, rows, meta, namespace in [("train", training, tm, 2800), ("valid", valid, vm, 2900)]:
        extra = Counter()
        for i in range(64):
            obs, _target, _ = fixture(namespace * 100000 + i)
            try:
                answer = analyze(obs, max_nodes=2000)
            except SearchLimit:
                extra["limit_skipped"] += 1
                continue
            d = answer.distance if answer.distance is not None else 15
            rows.append(pack(obs, None, distance=d))
            extra[str(d)] += 1
        meta["exact_auxiliary_strata"] = dict(extra)
    keys = {hashlib.sha256(r[0].tobytes() + r[1].tobytes()).digest() for r in training}
    valid = [
        r for r in valid if hashlib.sha256(r[0].tobytes() + r[1].tobytes()).digest() not in keys
    ]
    save_rows(root / "train.npz", training)
    save_rows(root / "valid.npz", valid)
    source = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            Path("zimortal/analysis/completion.py"),
            Path("zimortal/belief/sampling.py"),
            Path("zimortal/training/rollout.py"),
        ]
    }
    meta = {
        "training": tm,
        "validation": vm,
        "distinct_train_states": len(training),
        "distinct_valid_states": len(valid),
        "source": source,
        "parent": args.resume,
        "files": {
            n: hashlib.sha256((root / n).read_bytes()).hexdigest()
            for n in ["train.npz", "valid.npz"]
        },
        "value_units": "rollout mixture of soft-policy Q: net currency/100; frozen continuation policy",
        "belief_label": "simulator deck counts; labels only, never input",
        "distance_label": "exact optimistic legal accepted intakes; limits skipped; 15=unreachable",
    }
    (root / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")


def uniform_belief(x):
    known = np.rint((x[:, 0] + x[:, 1]) * 4).astype(int)
    unknown = 4 - known
    wall = np.rint(x[:, 27, 0] * 20).astype(int)
    out = np.zeros((len(x), 20, 5))
    for i, u in enumerate(unknown):
        n = int(u.sum())
        k = int(wall[i])
        denominator = math.comb(n, k)
        for tile, count in enumerate(u):
            for v in range(5):
                if v <= count and 0 <= k - v <= n - count:
                    out[i, tile, v] = (
                        math.comb(int(count), v) * math.comb(n - int(count), k - v) / denominator
                    )
    return out


def metrics(model, rows):
    device = next(model.parameters()).device
    totals = Counter()
    with torch.no_grad():
        for start in range(0, len(rows), 128):
            r = rows[start : start + 128]
            x, a, m, target = minibatch(r, device)
            outputs, b, d = model.forward_planning(x, a, m)
            outputs = tuple(t.cpu() if t is not None else None for t in outputs)
            b, d, x, target = b.cpu(), d.cpu(), x.cpu(), target.cpu()
            idx = [i for i, t in enumerate(r) if t[4] is not None]
            if idx:
                truth = torch.tensor(np.stack([r[i][4] for i in idx]))
                prediction = b[idx].softmax(-1)
                totals["belief_tiles"] += truth.numel()
                totals["belief_correct"] += int((prediction.argmax(-1) == truth).sum())
                totals["belief_nll"] += float(
                    -prediction.gather(-1, truth[:, :, None]).clamp_min(1e-12).log().sum()
                )
                prior = torch.tensor(uniform_belief(x[idx].numpy()))
                totals["prior_correct"] += int((prior.argmax(-1) == truth).sum())
                totals["prior_nll"] += float(
                    -prior.gather(-1, truth[:, :, None]).clamp_min(1e-12).log().sum()
                )
                expected = (prediction * torch.arange(5)).sum(-1)
                totals["belief_count_mae"] += float((expected - truth).abs().sum())
            idx = [i for i, t in enumerate(r) if t[5] >= 0]
            if idx:
                truth = torch.tensor([r[i][5] for i in idx])
                totals["distance_labels"] += len(idx)
                totals["distance_correct"] += int((d[idx].argmax(-1) == truth).sum())
            idx = [i for i, t in enumerate(r) if t[2] is not None]
            if idx:
                totals["rollout_labels"] += len(idx)
                totals["policy_ce"] += float(-(target[idx] * outputs[0][idx].log_softmax(-1)).sum())
                totals["cash_value_mae"] += float(
                    (outputs[1][idx] * 100 - torch.tensor([r[i][3] * 100 for i in idx])).abs().sum()
                )
    return {
        "belief_tiles": totals["belief_tiles"],
        "belief_accuracy": totals["belief_correct"] / max(1, totals["belief_tiles"]),
        "belief_nll": totals["belief_nll"] / max(1, totals["belief_tiles"]),
        "uniform_unknown_allocation_accuracy": totals["prior_correct"]
        / max(1, totals["belief_tiles"]),
        "uniform_unknown_allocation_nll": totals["prior_nll"] / max(1, totals["belief_tiles"]),
        "belief_expected_count_mae": totals["belief_count_mae"] / max(1, totals["belief_tiles"]),
        "distance_labels": totals["distance_labels"],
        "distance_accuracy": totals["distance_correct"] / max(1, totals["distance_labels"]),
        "rollout_labels": totals["rollout_labels"],
        "rollout_policy_cross_entropy": totals["policy_ce"] / max(1, totals["rollout_labels"]),
        "cash_value_mae": totals["cash_value_mae"] / max(1, totals["rollout_labels"]),
    }


def atomic_save(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary)
    with temporary.open("rb") as saved:
        os.fsync(saved.fileno())
    os.replace(temporary, path)


def training_device(request):
    if request == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if request == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable; install CUDA PyTorch and check GPU access"
        )
    return request


def train(args):
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(99)
    rng = random.Random(99)
    root = Path(args.data)
    meta = json.loads((root / "manifest.json").read_text())
    for n, h in meta["files"].items():
        if hashlib.sha256((root / n).read_bytes()).hexdigest() != h:
            raise ValueError("dataset changed")
    training = read_rows(root / "train.npz")
    valid = read_rows(root / "valid.npz")
    device = training_device(args.device)
    parent = load_model(args.resume, device)
    model = PolicyValueNet(parent.architecture, parent.width, "huxi", "planning").to(device)
    model.load_state_dict(parent.state_dict(), strict=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    focus = [r for r in training if r[2] is not None or r[5] >= 0]
    logs = []
    best = float("inf")
    first_epoch = 0
    recovery = Path(args.output + ".recovery.pt")
    if args.continue_training:
        state = torch.load(recovery, map_location=device, weights_only=True)
        if (
            state["files"] != meta["files"]
            or state["parent_sha256"] != hashlib.sha256(Path(args.resume).read_bytes()).hexdigest()
        ):
            raise ValueError("resume dataset or parent changed")
        if (
            not Path(args.output).exists()
            or hashlib.sha256(Path(args.output).read_bytes()).hexdigest()
            != state["best_model_sha256"]
        ):
            raise ValueError("best checkpoint changed or missing")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        rng.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if device == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([t.cpu() for t in state["cuda_rng"]])
        first_epoch, best, logs = state["epoch"], state["best"], state["logs"]
    print(
        json.dumps(
            {
                "device": device,
                "torch": torch.__version__,
                "start_epoch": first_epoch,
                "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            }
        ),
        flush=True,
    )
    for epoch in range(first_epoch, args.epochs):
        model.train()
        order = training.copy()
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), 96):
            rows = order[start : start + 96] + rng.choices(focus, k=32)
            x, a, m, soft = minibatch(rows, device)
            outputs, belief, distance = model.forward_planning(x, a, m)
            with torch.no_grad():
                old = parent.forward_all(x, a, m)
            loss = 0.3 * nn.functional.kl_div(
                outputs[0].log_softmax(-1), old[0].softmax(-1), reduction="batchmean"
            ) + 0.1 * nn.functional.mse_loss(outputs[1], old[1])
            idx = [i for i, r in enumerate(rows) if r[2] is not None]
            if idx:
                loss += -(soft[idx] * outputs[0][idx].log_softmax(-1)).sum(-1).mean()
                loss += nn.functional.mse_loss(
                    outputs[1][idx], torch.tensor([rows[i][3] for i in idx], device=device)
                )
            idx = [i for i, r in enumerate(rows) if r[4] is not None]
            if idx:
                truth = torch.tensor(np.stack([rows[i][4] for i in idx]), device=device)
                loss += 0.3 * nn.functional.cross_entropy(
                    belief[idx].reshape(-1, 5), truth.flatten()
                )
            idx = [i for i, r in enumerate(rows) if r[5] >= 0]
            if idx:
                loss += 0.2 * nn.functional.cross_entropy(
                    distance[idx], torch.tensor([rows[i][5] for i in idx], device=device)
                )
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite planning loss")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        validation = metrics(model, valid)
        score = (
            validation["rollout_policy_cross_entropy"]
            + 0.1 * validation["belief_nll"]
            + 0.1 * (1 - validation["distance_accuracy"])
        )
        if score < best:
            best = score
            save_model(
                model,
                args.output,
                value_target="rollout soft-policy net payments/100 with frozen continuation",
                epoch=epoch + 1,
                training_device=device,
                torch_version=str(torch.__version__),
            )
        log = {"epoch": epoch + 1, "loss": sum(losses) / len(losses), "validation": validation}
        logs.append(log)
        atomic_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best": best,
                "logs": logs,
                "files": meta["files"],
                "parent_sha256": hashlib.sha256(Path(args.resume).read_bytes()).hexdigest(),
                "best_model_sha256": hashlib.sha256(Path(args.output).read_bytes()).hexdigest(),
                "python_rng": rng.getstate(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else None,
            },
            recovery,
        )
        print(json.dumps(log), flush=True)
        Path(args.report).write_text(
            json.dumps({"status": "running", "device": device, "epochs": logs}, indent=2) + "\n"
        )
    report = {
        "manifest": meta,
        "device": device,
        "torch": torch.__version__,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
        "epochs": logs,
        "checkpoint": args.output,
        "sha256": hashlib.sha256(Path(args.output).read_bytes()).hexdigest(),
        "selected_epoch": torch.load(args.output, map_location="cpu", weights_only=True)[
            "metadata"
        ]["epoch"],
        "validation": metrics(load_model(args.output), valid),
        "rollout_uncertainty": "finite MC standard errors, no optimal-EV claim",
    }
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")


def evaluate(args):
    torch.set_num_threads(2)
    models = {"baseline": args.resume, "planning": args.output}
    report = {"models": {}}
    for name, path in models.items():
        model = load_model(path)
        report["models"][name] = {
            "checkpoint": path,
            "random": tournament(model, list(range(35000, 35050)), include_games=True),
            "teacher": tournament(
                model, list(range(35500, 35550)), opponent="teacher", include_games=True
            ),
        }
        print(
            json.dumps(
                {
                    "model": name,
                    "random": report["models"][name]["random"]["mean_payoff"],
                    "teacher": report["models"][name]["teacher"]["mean_payoff"],
                }
            ),
            flush=True,
        )
    from .boundary_review import paired_interval

    report["paired"] = {
        k: paired_interval(
            report["models"]["baseline"][k]["game_results"],
            report["models"]["planning"][k]["game_results"],
        )
        for k in ("random", "teacher")
    }
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["build", "train", "evaluate"])
    p.add_argument("--data", default="data/generated/planning-pilot")
    p.add_argument("--resume", default="checkpoints/huxi-resnet.pt")
    p.add_argument("--output", default="checkpoints/planning-resnet.pt")
    p.add_argument("--report", default="docs/training/planning-pilot.json")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--roots", type=int, default=32)
    p.add_argument("--rollouts", type=int, default=16)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--continue-training", action="store_true")
    args = p.parse_args()
    {"build": build, "train": train, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
