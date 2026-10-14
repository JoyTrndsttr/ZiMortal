"""Finite A/B/C pilot: 1000 audited v0 games, MC-only and MC+CQL; then STOP."""

import argparse
import copy
import json
import math
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from zimortal.model.mortal_q import MortalQNet

from .active import atomic_json
from .mortal_data import audit_dataset, generate_dataset, load_rows
from .online_rl import atomic_checkpoint, pack, source_hashes
from .runtime import load_model, save_model

BASELINE = "data/generated/league-v2/stage-02/best-dev.pt"


def mc_target(cash, steps, gamma=1.0):
    if not math.isfinite(gamma) or not 0 < gamma <= 1 or (steps < 0).any():
        raise ValueError("invalid discount or remaining own decisions")
    return cash * gamma**steps


def q_loss(q, chosen, target, mask, alpha=0.0):
    if not math.isfinite(alpha) or alpha < 0 or not mask.any(-1).all():
        raise ValueError("invalid CQL coefficient/mask")
    if not mask.gather(1, chosen[:, None]).all():
        raise ValueError("dataset chosen action must be legal")
    selected = q.gather(1, chosen[:, None]).squeeze(-1)
    mc = F.smooth_l1_loss(selected, target)
    cql = (q.masked_fill(~mask, -torch.inf).logsumexp(-1) - selected).mean()
    return mc + alpha * cql, {"mc_loss": mc, "cql_loss": cql}


def tensors(rows, device, gamma):
    x, a, mask = pack([r["features"] for r in rows], [r["actions"] for r in rows], device)
    chosen = torch.tensor([int(r["chosen"]) for r in rows], device=device)
    cash = torch.tensor([float(r["terminal_cash"]) for r in rows], device=device)
    steps = torch.tensor([int(r["steps_to_terminal"]) for r in rows], device=device)
    return x, a, mask, chosen, mc_target(cash, steps, gamma)


def regression(model, rows, *, device="cpu", gamma=1.0, batch_size=256, gate_threshold=None):
    model.eval()
    predictions, targets, legal_q = [], [], []
    changed = {}
    q_rows = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            x, a, m, chosen, target = tensors(batch, device, gamma)
            q = model.forward_q(x, a, m)[0]
            if not torch.isfinite(q[m]).all():
                raise RuntimeError("nonfinite legal Q")
            predictions.extend(q.gather(1, chosen[:, None]).squeeze(-1).cpu().tolist())
            targets.extend(target.cpu().tolist())
            legal_q.extend(q[m].cpu().tolist())
            q_rows.extend(
                [v[: len(r["actions"])].cpu().numpy() for v, r in zip(q, batch, strict=True)]
            )
            for r, choice in zip(batch, q.argmax(-1).cpu().tolist(), strict=True):
                entry = changed.setdefault(str(r["family"]), {"decisions": 0, "changed": 0})
                entry["decisions"] += 1
                entry["changed"] += choice != int(r["parent_greedy"])
    from .mortal_diagnostics import diagnostics

    p, t = np.asarray(predictions), np.asarray(targets)
    return {
        **diagnostics(
            rows,
            p,
            t,
            q_rows,
            thresholds=(0, 0.02, 0.05, 0.1, 0.2) if gate_threshold is None else (gate_threshold,),
        ),
        "decisions": len(rows),
        "mae": float(np.mean(abs(p - t))),
        "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
        "correlation": float(np.corrcoef(p, t)[0, 1])
        if p.std() > 1e-10 and t.std() > 1e-10
        else None,
        "q_mean": float(np.mean(legal_q)),
        "q_std": float(np.std(legal_q)),
        "q_min": float(min(legal_q)),
        "q_max": float(max(legal_q)),
        "target_mean": float(t.mean()),
        "target_std": float(t.std()),
        "action_reranking": changed,
        "changed_action_rate": sum(v["changed"] for v in changed.values()) / len(rows),
    }


def fit(
    initial,
    train,
    dev,
    test,
    directory,
    *,
    device="cpu",
    alpha=0.0,
    gamma=1.0,
    epochs=8,
    batch_size=256,
    seed=1414,
    patience=None,
    gated_diagnostic=False,
    n_step=0,
    target_sync_updates=250,
    evaluate_test=True,
):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    model = copy.deepcopy(initial).to(device)
    from .mortal_sarsa import HardTarget, future_indices, parameter_hash, sarsa_target

    if n_step < 0:
        raise ValueError("nonnegative own-decision horizon required")
    initial_hash = parameter_hash(model)
    target_net = HardTarget(model, target_sync_updates) if n_step else None
    links = future_indices(train, n_step) if n_step else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(seed)
    initial_train = regression(model, train, device=device, gamma=gamma)
    initial_dev = regression(model, dev, device=device, gamma=gamma)
    best = float("inf")
    logs = []
    stale, best_epoch = 0, 0
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        totals = {"mc_loss": 0.0, "cql_loss": 0.0}
        count = 0
        norms = []
        batch_order = __import__("hashlib").sha256()
        bootstrap_counts = 0
        target_values = []
        for indices in np.array_split(
            rng.permutation(len(train)), math.ceil(len(train) / batch_size)
        ):
            batch_order.update(indices.astype(np.int64).tobytes())
            batch = [train[i] for i in indices]
            x, a, m, chosen, target = tensors(batch, device, gamma)
            if target_net is not None:
                target = sarsa_target(
                    train, indices, links, target_net.model, n=n_step, device=device, gamma=gamma
                )
                bootstrap_counts += int((links[indices] >= 0).sum())
            target_values.extend(target.detach().cpu().tolist())
            q = model.forward_q(x, a, m)[0]
            loss, parts = q_loss(q, chosen, target, m, alpha)
            if not torch.isfinite(loss) or not torch.isfinite(q[m]).all():
                raise RuntimeError("Q loss nonfinite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError("Q gradient nonfinite")
            norms.append(float(norm))
            optimizer.step()
            if target_net is not None:
                target_net.advance(model)
            for k, v in parts.items():
                totals[k] += float(v.detach()) * len(batch)
            count += len(batch)
        validation = regression(model, dev, device=device, gamma=gamma)
        log = {
            "epoch": epoch + 1,
            "batch_order_sha256": batch_order.hexdigest(),
            "training_target_mean": float(np.mean(target_values)),
            "training_target_std": float(np.std(target_values)),
            "bootstrap_rows": bootstrap_counts,
            "terminal_rows": count - bootstrap_counts,
            "target_sync_history": target_net.sync_history.copy() if target_net else [],
            "training_huber_loss": totals["mc_loss"] / count,
            "gradient_norm_mean": float(np.mean(norms)),
            "gradient_norm_max": max(norms),
            **{k: v / count for k, v in totals.items()},
            "wait_loss": 0,
            "huxi_loss": 0,
            "auxiliary_heads_enabled": False,
            "development": validation,
        }
        logs.append(log)
        atomic_json(root / "epochs.json", logs)
        if validation["rmse"] < best:
            best = validation["rmse"]
            stale, best_epoch = 0, epoch + 1
            save_model(
                model,
                root / "best-dev.pt",
                target="terminal payments[seat]/100"
                if not n_step
                else f"{n_step}-step behavior SARSA cash/100",
                n_step=n_step,
                target_sync_updates=target_sync_updates if n_step else None,
                gamma=gamma,
                alpha_cql=alpha,
            )
            atomic_checkpoint(
                root / "best-optimizer.pt",
                {
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "checkpoint_sha256": __import__("hashlib")
                    .sha256((root / "best-dev.pt").read_bytes())
                    .hexdigest(),
                    "target_network": target_net.model.state_dict() if target_net else None,
                    "target_updates": target_net.updates if target_net else 0,
                    "target_sync_updates": target_net.interval if target_net else None,
                    "target_sync_history": target_net.sync_history.copy() if target_net else [],
                    "numpy_rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
                },
            )
        else:
            stale += 1
        atomic_json(
            root.parent / "progress.json",
            {"status": "training", "variant": root.name, "epoch": epoch + 1, "epochs": epochs},
        )
        if patience is not None and stale >= patience:
            break
    selected = load_model(root / "best-dev.pt", device)
    selected_dev = regression(selected, dev, device=device, gamma=gamma)
    from .mortal_diagnostics import select_gate

    threshold = select_gate(selected_dev) if gated_diagnostic else None
    result = {
        "alpha_cql": alpha,
        "n_step": n_step,
        "initial_parameter_sha256": initial_hash,
        "target_sync_updates": target_sync_updates if n_step else None,
        "loss_meaning": "Huber against terminal MC"
        if not n_step
        else "Huber against frozen behavior SARSA target; mc_loss legacy key is not MC regression",
        "evaluation_target": "realized terminal cash, never TD bootstrap",
        "gamma": gamma,
        "parameters": sum(p.numel() for p in model.parameters()),
        "initial_train": initial_train,
        "initial_dev": initial_dev,
        "best_epoch": best_epoch,
        "epochs_run": len(logs),
        "early_stopped": len(logs) < epochs,
        "selected_dev": selected_dev,
        "selected_train": regression(selected, train, device=device, gamma=gamma),
        "heldout": regression(selected, test, device=device, gamma=gamma, gate_threshold=threshold)
        if evaluate_test
        else None,
        "seconds": time.perf_counter() - started,
        "epochs": logs,
        "cash_strength_evaluated": False,
        "checkpoint": str(root / "best-dev.pt"),
    }
    atomic_json(root / "result.json", result)
    return result


class UtilizationSampler:
    def __init__(self):
        self.stop = threading.Event()
        self.samples = []
        self.binary = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.is_set():
            try:
                p = subprocess.run(
                    [
                        self.binary,
                        "--query-gpu=utilization.gpu,utilization.memory",
                        "--format=csv,noheader,nounits",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                self.samples.append(
                    {
                        "time": time.time(),
                        "phase": self.phase,
                        "gpu_query": p.stdout.strip(),
                        "error": p.stderr.strip(),
                    }
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                self.samples.append({"error": str(exc)})
            self.stop.wait(0.5)

    def __enter__(self):
        self.phase = "data"
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=3)


def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    root = Path(args.directory)
    root.mkdir(parents=True, exist_ok=False)
    from .online_rl import digest

    config = {
        **vars(args),
        "source_hashes": source_hashes(),
        "baseline_sha256": digest(args.baseline),
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
    }
    atomic_json(root / "config.json", config)
    baseline = load_model(args.baseline, args.device)
    baseline.requires_grad_(False)
    model = MortalQNet(width=args.width, latent=args.latent).to(args.device)
    with torch.inference_mode():
        test_mask = torch.tensor([[True, False, True], [True, True, True]], device=args.device)
        test_q = model.forward_q(
            torch.randn(2, 55, 20, device=args.device),
            torch.randn(2, 3, model.action_encoder[0].in_features, device=args.device),
            test_mask,
        )[0]
        assert torch.isfinite(test_q[test_mask]).all()
        assert torch.isneginf(test_q[~test_mask]).all()
    with UtilizationSampler() as sampler:
        data_started = time.perf_counter()
        manifest = generate_dataset(
            baseline,
            root / "data",
            games=args.games,
            start=args.start,
            shard_games=256,
            device=args.device,
            slots=args.slots,
            temperature=args.temperature,
            epsilon=args.epsilon,
            top_p=args.top_p,
            metadata=config,
        )
        data_pipeline_seconds = time.perf_counter() - data_started
        audit_started = time.perf_counter()
        audit = audit_dataset(root / "data", manifest)
        audit["seconds"] = time.perf_counter() - audit_started
        atomic_json(root / "data-audit.json", audit)
        rows = load_rows(root / "data", manifest)
        # Entire games held out; no related learner states cross splits.
        train = [r for r in rows if int(r["seed"]) % 10 >= 2]
        dev = [r for r in rows if int(r["seed"]) % 10 == 1]
        test = [r for r in rows if int(r["seed"]) % 10 == 0]
        if not train or not dev or not test:
            raise RuntimeError("empty game-level split")
        if (
            source_hashes() != config["source_hashes"]
            or digest(args.baseline) != config["baseline_sha256"]
        ):
            raise RuntimeError("producer/baseline changed")
        sampler.phase = "training"
        results = {}
        for name, alpha in [("mc", 0.0), ("mc-cql", args.alpha_cql)]:
            results[name] = fit(
                model,
                train,
                dev,
                test,
                root / name,
                device=args.device,
                alpha=alpha,
                gamma=args.gamma,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=args.seed,
            )
    profile = manifest["profiles"]
    seconds = sum(p["wall_seconds"] for p in profile)
    completed = sum(s["games"] for s in manifest["shards"])
    result = {
        "status": "complete_stage_ABC_stop",
        "completed_games": completed,
        "decisions": len(rows),
        "excluded_games": len(manifest["failures"]),
        "game_split": {
            "train": len({int(r["seed"]) for r in train}),
            "dev": len({int(r["seed"]) for r in dev}),
            "test": len({int(r["seed"]) for r in test}),
        },
        "parameters": sum(p.numel() for p in model.parameters()),
        "data_audit": audit,
        "device": args.device,
        "gpu": config["gpu"],
        "data_seconds": data_pipeline_seconds,
        "simulation_seconds": seconds,
        "games_per_second": completed / data_pipeline_seconds,
        "decisions_per_second": len(rows) / seconds,
        "profiles": profile,
        "utilization_samples": sampler.samples,
        "variants": results,
        "cash_strength_evaluated": False,
        "baseline_changed": False,
        "next": "report first; do not automatically start 10k/50k or online",
    }
    atomic_json(root / "result.json", result)
    atomic_json(root / "progress.json", {"status": result["status"], "completed_games": completed})
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("variants", "utilization_samples", "profiles")
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="data/generated/mortal-v1")
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--start", type=int, default=1000000000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--latent", type=int, default=256)
    parser.add_argument("--slots", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1414)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--alpha-cql", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=1.0)
    args = parser.parse_args()
    if (
        min(args.games, args.slots, args.epochs, args.batch_size) < 1
        or not math.isfinite(args.alpha_cql)
        or args.alpha_cql <= 0
    ):
        parser.error("positive counts and positive CQL comparison required")
    if (
        not 0 < args.gamma <= 1
        or not math.isfinite(args.temperature)
        or args.temperature <= 0
        or not 0 <= args.epsilon <= 1
        or not 0 < args.top_p <= 1
    ):
        parser.error("invalid discount/exploration")
    try:
        run(args)
    except Exception as exc:
        root = Path(args.directory)
        if root.exists():
            atomic_json(root / "progress.json", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
