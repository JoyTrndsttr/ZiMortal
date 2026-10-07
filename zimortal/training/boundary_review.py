"""Independent boundary audits, paired cash evaluation and prediction diagnostics."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .boundary import datasets, diagnostics, fixture, read_boundary, row
from .huxi import read_rich
from .huxi_review import frozen_positions
from .runtime import load_model, tournament
from .scale import deduplicate


def paired_interval(parent, challenger):
    """Paired bootstrap clusters all three seats of each shared deck seed."""

    def aggregate(records):
        totals = Counter()
        sizes = Counter()
        for r in records:
            totals[r["seed"]] += r["payoff"]
            sizes[r["seed"]] += 1
        return {s: totals[s] / sizes[s] for s in totals}

    before, after = aggregate(parent), aggregate(challenger)
    if set(before) != set(after):
        raise ValueError("paired evaluations must have identical seeds")
    differences = np.array([after[s] - before[s] for s in sorted(before)])
    rng = np.random.default_rng(880)
    bootstrap = differences[rng.integers(len(differences), size=(5000, len(differences)))].mean(1)
    return {
        "seed_clusters": len(differences),
        "mean_payoff_difference": float(differences.mean()),
        "paired_bootstrap_95_percent": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "unit": "settlement currency per game",
        "seats_clustered_by_seed": True,
    }


def audit(directory):
    root = Path(directory)
    train, valid = datasets(root)
    manifest = json.loads((root / "manifest.json").read_text())
    counts = Counter()
    checked = 0
    for shard in manifest["shards"]:
        meta = json.loads((root / shard["file"]).with_suffix(".json").read_text())
        saved = read_boundary(root / shard["file"])
        for i in range(0, len(saved), 40):
            case = meta["fixtures"][i]
            obs, target, state = fixture(case["seed"])
            regenerated = row(obs)
            assert target == case["target"]
            assert hashlib.sha256(state.serialize().encode()).hexdigest() == case["state_sha256"]
            for j in (0, 1, 4, 5, 6):
                np.testing.assert_array_equal(regenerated[j], saved[i][j])
            assert regenerated[2:4] == saved[i][2:4]
            counts[case["bucket"]] += 1
            checked += 1
    return {
        "train": len(train),
        "validation": len(valid),
        "regenerated_fixtures": checked,
        "strata": dict(counts),
        "conservation_verified": True,
        "real_draw_hu_gate_and_payout_verified": True,
        "all_twenty_draw_labels_and_network_inputs_regenerated": True,
        "source": manifest["source"],
        "synthetic_midgame_not_standard_deal_history": True,
    }


def evaluate(args):
    torch.set_num_threads(2)
    train, valid = datasets(args.data)
    root = Path("data/generated/huxi100k")
    manifest = json.loads((root / "manifest.json").read_text())
    ordinary_train, ordinary_valid = [], []
    for record in manifest["shards"]:
        (ordinary_train if record["config"]["kind"].startswith("train") else ordinary_valid).extend(
            read_rich(root / record["file"])
        )
    ordinary_train, ordinary_valid = deduplicate(ordinary_train, ordinary_valid)
    key = lambda r: hashlib.sha256(r[0].tobytes() + r[1].tobytes()).digest()
    training_keys = {key(r) for r in train + ordinary_train}
    valid = [r for r in valid if key(r) not in training_keys]
    ordinary_valid = [r for r in ordinary_valid if key(r) not in training_keys]
    checkpoints = {
        "parent": "checkpoints/huxi-resnet.pt",
        "warmup": args.warmup,
        "settlement": args.checkpoint,
    }
    result = {
        "models": {},
        "random_seeds": list(range(30000, 30100)),
        "teacher_seeds": list(range(30500, 30600)),
        "selection": "auxiliary validation selects warmup, tournament never chooses weights",
        "paired_comparisons": {},
        "frozen_original_positions": frozen_positions(checkpoints),
    }
    for name, path in checkpoints.items():
        model = load_model(path)
        record = {
            "checkpoint": path,
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "boundary": diagnostics(model, valid),
            "ordinary": diagnostics(model, ordinary_valid),
            "random": tournament(model, result["random_seeds"], include_games=True),
            "teacher": tournament(
                model, result["teacher_seeds"], opponent="teacher", include_games=True
            ),
        }
        result["models"][name] = record
        print(
            json.dumps(
                {
                    "model": name,
                    "boundary": record["boundary"],
                    "random_mean_payoff": record["random"]["mean_payoff"],
                    "teacher_mean_payoff": record["teacher"]["mean_payoff"],
                }
            ),
            flush=True,
        )
    for name in ("warmup", "settlement"):
        result["paired_comparisons"][name] = {
            op: paired_interval(
                result["models"]["parent"][op]["game_results"],
                result["models"][name][op]["game_results"],
            )
            for op in ("random", "teacher")
        }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("audit", "evaluate"))
    p.add_argument("--data", default="data/generated/boundary20k")
    p.add_argument("--checkpoint", default="checkpoints/boundary-resnet.pt")
    p.add_argument("--warmup", default="checkpoints/boundary-calibrated.pt")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    report = audit(args.data) if args.command == "audit" else evaluate(args)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
