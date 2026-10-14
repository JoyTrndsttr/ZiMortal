"""One fixed independent replication of provisional counterfactual signs; no training."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import torch

from zimortal.engine import GameState, RuleClarificationRequired, RuleEngine

from .active import atomic_json
from .mortal_counterfactual import FAMILIES, paired_rollouts, paired_statistics
from .mortal_data import write_bytes
from .mortal_offline import BASELINE
from .online_rl import digest, source_hashes
from .runtime import load_model


def comparison(first, second):
    if first["resolved_sign"] == "unresolved":
        raise ValueError("replication cohort must have a provisional resolved sign")
    return {
        "same_point_sign": first["sign"] == second["sign"],
        "same_ci_sign": first["resolved_sign"] == second["resolved_sign"],
        "opposite_ci_sign": second["resolved_sign"] not in ("unresolved", first["resolved_sign"]),
    }


def run(args):
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    root, source = Path(args.directory), Path(args.source)
    root.mkdir(parents=True, exist_ok=False)
    original = json.loads((source / "config.json").read_text())
    inputs = {
        str(source / "roots.json"): digest(source / "roots.json"),
        **original["checkpoint_sha256"],
    }
    if digest(BASELINE) != original["checkpoint_sha256"][BASELINE]:
        raise RuntimeError("baseline changed")
    records = json.loads((source / "roots.json").read_text())
    selected = [r for r in records if r["stats"]["resolved_sign"] != "unresolved"]
    config = {
        "source": str(source),
        "input_sha256": inputs,
        "source_hashes": source_hashes(),
        "selection": "ALL first-batch bootstrap intervals excluding zero; positive and negative included",
        "rollouts": 32,
        "seed_namespace": "confirmation-v1",
        "not_for_training": True,
        "selected": len(selected),
    }
    atomic_json(root / "config.json", config)
    model = load_model(BASELINE, "cuda").requires_grad_(False)
    results, failures = [], []
    for index, row in enumerate(selected):
        path = source / row["evidence"]
        if digest(path) != row["sha256"]:
            raise RuntimeError("first-batch evidence changed")
        with gzip.open(path, "rt") as file:
            evidence = json.load(file)
        state = GameState.deserialize(evidence["root"])
        legal = RuleEngine().legal_actions(state)
        seed = int.from_bytes(
            hashlib.sha256(f"confirmation-v1:{row['id']}".encode()).digest()[:4], "little"
        )
        try:
            outcomes, pairs = paired_rollouts(
                state,
                legal[row["parent"]],
                legal[row["chosen"]],
                model,
                count=32,
                seed=seed,
                device="cuda",
            )
            second = paired_statistics(outcomes, seed)
            result = {
                "id": row["id"],
                "family": row["family"],
                "direction": row["direction"],
                "first": row["stats"],
                "second": second,
                "comparison": comparison(row["stats"], second),
                "outcomes": outcomes.tolist(),
            }
            sha = write_bytes(
                root / f"root-{index:04d}.json.gz",
                gzip.compress(
                    json.dumps(
                        {"root": state.serialize(), "result": result, "pairs": pairs}
                    ).encode(),
                    mtime=0,
                ),
            )
            results.append({**result, "sha256": sha, "evidence": f"root-{index:04d}.json.gz"})
        except RuleClarificationRequired as exc:
            failures.append(
                {"id": row["id"], "error": str(exc), "whole_root_excluded": True, "seed": seed}
            )
        atomic_json(
            root / "progress.json",
            {
                "status": "independent_confirmation",
                "done": index + 1,
                "total": len(selected),
                "valid": len(results),
                "failed": len(failures),
            },
        )

    def group_summary(group):
        return {
            "roots": len(group),
            "same_point_sign": sum(r["comparison"]["same_point_sign"] for r in group),
            "same_ci_sign": sum(r["comparison"]["same_ci_sign"] for r in group),
            "opposite_ci_sign": sum(r["comparison"]["opposite_ci_sign"] for r in group),
            "second_unresolved": sum(r["second"]["resolved_sign"] == "unresolved" for r in group),
        }

    if source_hashes() != config["source_hashes"] or any(
        digest(p) != sha for p, sha in inputs.items()
    ):
        raise RuntimeError("frozen inputs/source changed")
    result = {
        "status": "complete_independent_confirmation_stop",
        "selected": len(selected),
        "valid": len(results),
        "failures": failures,
        "overall": group_summary(results),
        "families": {f: group_summary([r for r in results if r["family"] == f]) for f in FAMILIES},
        "first_signs": {
            sign: group_summary([r for r in results if r["first"]["resolved_sign"] == sign])
            for sign in ("positive", "negative")
        },
        "not_for_training": True,
    }
    atomic_json(root / "roots.json", results)
    atomic_json(root / "result.json", result)
    atomic_json(root / "progress.json", {"status": result["status"]})
    print(result["status"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/generated/mortal-counterfactual-v1")
    parser.add_argument("--directory", default="data/generated/mortal-counterfactual-confirm-v1")
    args = parser.parse_args()
    if Path(args.directory).exists():
        parser.error("new directory required")
    try:
        run(args)
    except Exception as exc:
        if Path(args.directory).exists():
            atomic_json(
                Path(args.directory) / "progress.json", {"status": "failed", "error": str(exc)}
            )
        raise


if __name__ == "__main__":
    main()
