"""Read-only post-lock analysis for the matched-update data-volume experiment."""

import json
from pathlib import Path

import torch

from .active import atomic_json
from .mortal_faithful_diagnostics import raw_diagnostics
from .online_rl import digest


def run():
    root = Path("data/generated/mortal-volume20k-v2")
    result = json.loads((root / "result.json").read_text())
    lock = json.loads((root / "selection-lock.json").read_text())
    common = lock["common_update"]
    proof = {}
    for name, sha in lock["checkpoints"].items():
        path = root / name / "selected-common.pt"
        if digest(path) != sha:
            raise RuntimeError("selected checkpoint changed")
        model = torch.load(path, map_location="cpu", weights_only=True)
        optimizer = torch.load(
            root / name / "selected-common-optimizer.pt", map_location="cpu", weights_only=True
        )
        if model["metadata"]["updates"] != common or optimizer["updates"] != common:
            raise RuntimeError("selected update mismatch")
        steps = {int(v["step"]) for v in optimizer["optimizer"]["state"].values()}
        if steps != {common}:
            raise RuntimeError("actual optimizer step counters mismatch")
        if optimizer["checkpoint_sha256"] != sha:
            raise RuntimeError("selected optimizer does not pair with model")
        f = result["fits"][name]
        if f["updates"] != 5200 or f["examples_seen"] != 5200 * 256 or f["dev_checks"] != 20:
            raise RuntimeError("budget/evaluation-frequency mismatch")
        proof[name] = {
            "updates": f["updates"],
            "selected_update": common,
            "dev_checks": f["dev_checks"],
            "examples_seen": f["examples_seen"],
        }
    rows = {
        name: raw_diagnostics(root / "screening" / f"decisions-{name}-raw.jsonl.gz")
        for name in result["fits"]
    }
    gap = {
        name: {
            "common_update": common,
            "selected_train_rmse": f["common_selected_train"]["rmse"],
            "selected_dev_rmse": f["common_selected_dev"]["rmse"],
            "selected_gap": f["common_selected_dev"]["rmse"] - f["common_selected_train"]["rmse"],
            "final_train_rmse": f["logs"][-1]["training"]["rmse"],
            "final_dev_rmse": f["logs"][-1]["development"]["rmse"],
            "final_gap": f["logs"][-1]["train_dev_rmse_gap"],
        }
        for name, f in result["fits"].items()
    }
    raw = result["B_minus_A"]["raw"]["0"]
    guarded = result["B_minus_A"]["guarded"]["0"]
    raw_positive = raw["paired_seed_bootstrap_95"][0] > 0
    guard_nonnegative = guarded["paired_seed_bootstrap_95"][0] >= 0
    atomic_json(
        root / "review.json",
        {
            "fairness_proof": proof,
            "train_dev_gap": gap,
            "raw_trajectory_diagnostics": rows,
            "raw_significant_improvement": raw_positive,
            "guard_nonregression_proven": guard_nonnegative,
            "recommend50k": raw_positive and guard_nonnegative,
            "no_extra_test_selection": True,
            "online_started": False,
        },
    )


if __name__ == "__main__":
    run()
