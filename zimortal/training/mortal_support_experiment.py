"""Finite CQL/support dev selection, locked test and independent cash screening."""

import argparse
import copy
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from zimortal.model.mortal_q import MortalQNet

from .active import atomic_json
from .mortal_credit import audit_screening
from .mortal_data import audit_dataset, load_rows
from .mortal_offline import BASELINE, fit, regression
from .mortal_support import SupportPolicy, summarize
from .mortal_v2 import game_split
from .online_rl import digest, pack, paired_summary, play, source_hashes
from .runtime import load_model

THRESHOLDS = (0.01, 0.03, 0.05, 0.1)


def offline_support(q, rows, threshold):
    records = []
    with torch.inference_mode():
        for start in range(0, len(rows), 256):
            batch = rows[start : start + 256]
            scores = (
                q.forward_q(
                    *pack([r["features"] for r in batch], [r["actions"] for r in batch], "cuda")
                )[0]
                .cpu()
                .numpy()
            )
            for r, values in zip(batch, scores, strict=True):
                p = np.asarray(r["behavior_probabilities"])
                parent = int(r["parent_greedy"])
                values = values[: len(p)]
                keep = np.ones(len(p), dtype=bool) if threshold is None else p > threshold
                keep[parent] = True
                chosen = int(np.where(keep, values, -np.inf).argmax())
                ranked = np.sort(values[keep])
                records.append(
                    {
                        "candidate_count": len(p),
                        "filtered": int((~keep).sum()),
                        "changed": chosen != parent,
                        "behavior_probability": float(p[chosen]),
                        "parent_probability": float(p[parent]),
                        "retained_mean_probability": float(p[keep].mean()),
                        "margin": float(ranked[-1] - ranked[-2]) if len(ranked) > 1 else None,
                        "terminal_cash": float(r["terminal_cash"]) * 100,
                    }
                )
    result = summarize(records)
    # Recorded return does not label counterfactual Q argmax.
    result["overrides"].pop("cash_by_probability")
    return result


def verify_behavior(parent, rows, config):
    from .mortal_support import behavior_probability

    # CPU FP32 reference avoids CUDA batch-size dependent TF32 kernel error.
    parent = copy.deepcopy(parent).cpu().eval()
    maximum = 0.0
    with torch.inference_mode():
        for start in range(0, len(rows), 256):
            batch = rows[start : start + 256]
            x, a, m = pack([r["features"] for r in batch], [r["actions"] for r in batch], "cpu")
            logits = parent(x, a, m)[0]
            p = behavior_probability(logits, a, m, config).cpu().numpy()
            choices = logits.argmax(-1).cpu().tolist()
            for r, v, c in zip(batch, p, choices, strict=True):
                error = float(np.max(abs(v[: len(r["actions"])] - r["behavior_probabilities"])))
                maximum = max(maximum, error)
                if c != int(r["parent_greedy"]) or error > 1e-5:
                    raise RuntimeError("stored behavior/greedy mismatch")
    return {"rows": len(rows), "maximum_probability_error": maximum}


def evaluate(root, policies, baseline, config, specs):
    root.mkdir(parents=True, exist_ok=True)
    outcomes, diagnostics = {}, {}
    for name, (q, threshold) in policies.items():
        policy = SupportPolicy(q, baseline, config, threshold).eval()
        records = []
        atomic_json(root / "progress.json", {"status": "cash", "policy": name, "games": len(specs)})
        with (
            gzip.open(root / f"screen-{name}-replays.jsonl.gz", "wt") as replay,
            gzip.open(root / f"decisions-{name}.jsonl.gz", "wt") as decisions,
        ):

            def sink(e):
                replay.write(json.dumps(e) + "\n")

            def decision_sink(batch, records=records):
                for record in batch:
                    decisions.write(json.dumps(record) + "\n")
                    records.append(
                        {
                            k: v
                            for k, v in record.items()
                            if k not in ("q", "actions", "probabilities")
                        }
                    )

            _, games, errors = play(
                policy,
                [baseline],
                specs,
                device="cuda",
                slots=32,
                training=False,
                replay_sink=sink,
                decision_recorder=policy.record,
                decision_sink=decision_sink,
            )
        if errors or len(games) != len(specs):
            atomic_json(root / "failures.json", errors)
            raise RuntimeError("incomplete cash pairs")
        outcomes[name] = games
        diagnostics[name] = summarize(records)
        atomic_json(root / "screening-outcomes.json", outcomes)
        atomic_json(root / "diagnostics.json", diagnostics)
    audit = audit_screening(root)
    return {
        "audit": audit,
        "diagnostics": diagnostics,
        "comparisons": {
            name: paired_summary(g, outcomes["baseline"], 1414) for name, g in outcomes.items()
        },
    }, outcomes


def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(1414)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    root = Path(args.directory)
    root.mkdir(parents=True, exist_ok=False)
    data = Path("data/generated/mortal-v2/data")
    manifest = json.loads((data / "manifest.json").read_text())
    original = json.loads((data.parent / "config.json").read_text())
    if digest(BASELINE) != original["baseline_sha256"]:
        raise RuntimeError("baseline changed")
    for path, sha in original["source_hashes"].items():
        if path.startswith(("zimortal/engine/", "zimortal/model/")) and digest(path) != sha:
            raise RuntimeError("representation/engine changed")
    config = {
        "source_hashes": source_hashes(),
        "baseline_sha256": digest(BASELINE),
        "manifest_sha256": digest(data / "manifest.json"),
        "family_config": original["family_config"],
        "alpha_grid": [0, 0.05, 0.1],
        "threshold_grid": THRESHOLDS,
        "dev_cash_seeds": 200,
        "test_cash_start": 1040000000,
        "test_cash_seeds": 500,
        "selection": "alpha maximizes dev raw mean cash; common threshold maximizes pooled raw/CQL support dev mean cash; ties prefer higher threshold then smaller alpha",
        "seed": 1414,
        "epochs": 20,
        "patience": 4,
        "batch_size": 256,
        "optimizer": "AdamW lr3e-4",
        "n_step": 3,
        "target_sync_updates": 250,
        "gamma": 1.0,
        "device": "cuda",
        "torch": str(torch.__version__),
    }
    atomic_json(root / "config.json", config)
    baseline = load_model(BASELINE, "cuda").requires_grad_(False)
    initial = MortalQNet().cuda()
    audit = audit_dataset(data, manifest)
    if audit["games"] != 10000:
        raise RuntimeError("exact 10k required")
    atomic_json(root / "data-audit.json", audit)
    rows = load_rows(data, manifest)
    train, dev, test = game_split(rows)
    atomic_json(
        root / "behavior-audit.json", verify_behavior(baseline, rows, config["family_config"])
    )
    fits, models = {}, {}
    for name, alpha in (("raw", 0.0), ("cql005", 0.05), ("cql01", 0.1)):
        fits[name] = fit(
            initial,
            train,
            dev,
            [],
            root / name,
            device="cuda",
            alpha=alpha,
            n_step=3,
            epochs=20,
            patience=4,
            evaluate_test=False,
        )
        models[name] = load_model(fits[name]["checkpoint"], "cuda").requires_grad_(False)
    if len({v["initial_parameter_sha256"] for v in fits.values()}) != 1:
        raise RuntimeError("init mismatch")
    for i in range(min(v["epochs_run"] for v in fits.values())):
        if len({v["epochs"][i]["batch_order_sha256"] for v in fits.values()}) != 1:
            raise RuntimeError("batch mismatch")
    dev_seeds = sorted({int(r["seed"]) for r in dev})[:200]
    specs = [(seed, seed % 3, seat, 0) for seed in dev_seeds for seat in range(3)]
    atomic_json(root / "dev-seeds.json", dev_seeds)
    alpha_result, alpha_out = evaluate(
        root / "dev-alpha",
        {"baseline": (None, None), **{k: (v, None) for k, v in models.items()}},
        baseline,
        config["family_config"],
        specs,
    )
    cql = max(
        ("cql005", "cql01"),
        key=lambda k: (np.mean([r["cash"] for r in alpha_out[k]]), -fits[k]["alpha_cql"]),
    )
    grid = {"baseline": (None, None)}
    for t in THRESHOLDS:
        grid[f"raw-{t}"] = (models["raw"], t)
        grid[f"cql-{t}"] = (models[cql], t)
    support_result, support_out = evaluate(
        root / "dev-support", grid, baseline, config["family_config"], specs
    )
    threshold = max(
        THRESHOLDS,
        key=lambda t: (
            np.mean([r["cash"] for name in (f"raw-{t}", f"cql-{t}") for r in support_out[name]]),
            t,
        ),
    )
    lock = {
        "cql_variant": cql,
        "alpha": fits[cql]["alpha_cql"],
        "support_threshold": threshold,
        "dev_alpha": alpha_result,
        "dev_support": support_result,
        "checkpoints": {v["checkpoint"]: digest(v["checkpoint"]) for v in fits.values()},
        "test_not_evaluated": True,
    }
    atomic_json(root / "selection-lock.json", lock)
    policies = {
        "baseline": (None, None),
        "raw": (models["raw"], None),
        "cql": (models[cql], None),
        "support": (models["raw"], threshold),
        "cql-support": (models[cql], threshold),
    }
    offline = {}
    for name, (q, t) in policies.items():
        if q is None:
            continue
        offline[name] = {
            "chosen_action": regression(q, test, device="cuda", gate_threshold=0.0),
            "selection": offline_support(q, test, t),
        }
    atomic_json(root / "heldout.json", offline)
    specs = [
        (seed, seed % 3, seat, 0) for seed in range(1040000000, 1040000500) for seat in range(3)
    ]
    cash, out = evaluate(root / "screening", policies, baseline, config["family_config"], specs)
    cash["vs_raw"] = {name: paired_summary(g, out["raw"], 1414) for name, g in out.items()}
    if (
        source_hashes() != config["source_hashes"]
        or digest(BASELINE) != config["baseline_sha256"]
        or digest(data / "manifest.json") != config["manifest_sha256"]
        or any(digest(p) != sha for p, sha in lock["checkpoints"].items())
    ):
        raise RuntimeError("frozen source/checkpoint changed")
    for shard in manifest["shards"]:
        if (
            digest(data / shard["data"]) != shard["sha256"]
            or digest(data / shard["replays"]) != shard["replay_sha256"]
        ):
            raise RuntimeError("data changed")
    result = {
        "status": "complete_support_experiment_stop",
        "fits": fits,
        "selection": lock,
        "heldout": offline,
        "screening": cash,
        "baseline_changed": False,
    }
    atomic_json(root / "result.json", result)
    atomic_json(root / "progress.json", {"status": result["status"]})
    print(result["status"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="data/generated/mortal-support-v1")
    args = parser.parse_args()
    if Path(args.directory).exists():
        parser.error("new experiment directory required; never overwrite evidence")
    try:
        run(args)
    except Exception as exc:
        root = Path(args.directory)
        if root.exists():
            atomic_json(root / "progress.json", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
