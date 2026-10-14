"""Post-training diagnostic tables, CSV curves and conditional HU correlation."""

import csv
import gzip
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from .active import atomic_json
from .mortal_action_coverage import clustered_correlation_ci, template_coverage
from .mortal_data import load_rows
from .mortal_faithful_diagnostics import SOURCE, chosen_kind
from .mortal_sarsa import parameter_hash
from .mortal_v2 import game_split
from .online_rl import pack, paired_summary
from .runtime import load_model


def run():
    torch.set_num_threads(2)
    root = Path("data/generated/mortal-faithful-diagnostics-v1")
    result = json.loads((root / "result.json").read_text())
    hashes = {f["initial_parameter_sha256"] for f in result["fits"].values()}
    if len(hashes) != 1:
        raise RuntimeError("initialization differs across learning rates")
    for a, b in itertools.combinations(result["fits"].values(), 2):
        if any(
            x["batch_order_sha256"] != y["batch_order_sha256"]
            for x, y in zip(a["epochs"], b["epochs"])
        ):
            raise RuntimeError("shared epoch main batch order differs")
    comparisons = {}
    curves = []
    for name, fit in result["fits"].items():
        logs = fit["epochs"]
        best = logs[fit["best_epoch"] - 1]
        last = logs[-1]
        comparisons[name] = {
            "best_epoch": fit["best_epoch"],
            "epochs": len(logs),
            "best_dev_rmse_yuan": best["development"]["rmse"] * 100,
            "last_dev_rmse_yuan": last["development"]["rmse"] * 100,
            "best_train_rmse_yuan": best["training_evaluation"]["rmse"] * 100,
            "last_train_rmse_yuan": last["training_evaluation"]["rmse"] * 100,
            "best_training_loss": best["training_regression_loss"],
            "last_training_loss": last["training_regression_loss"],
            "dev_worse_epochs_after_best": sum(
                e["development"]["rmse"] > best["development"]["rmse"]
                for e in logs[fit["best_epoch"] :]
            ),
            "post_best_train_loss_decreases": sum(
                b["training_regression_loss"] < a["training_regression_loss"]
                for a, b in zip(logs[fit["best_epoch"] - 1 :], logs[fit["best_epoch"] :])
            ),
        }
        for e in logs:
            for split, key in [
                ("train", "training_evaluation"),
                ("dev", "development"),
                ("test", "test_diagnostic_only"),
            ]:
                s = e[key]
                curves.append(
                    {
                        "configuration": name,
                        "epoch": e["epoch"],
                        "split": split,
                        "training_half_mse": e["training_regression_loss"],
                        "cql_loss": e["cql_loss"],
                        "mae_yuan": s["mae"] * 100,
                        "rmse_yuan": s["rmse"] * 100,
                        "corr": s["correlation"],
                        "chosen_q_std_yuan": s["chosen_q_std"] * 100,
                        "all_legal_q_std_yuan": s["q_std"] * 100,
                        "target_std_yuan": s["target_std"] * 100,
                        "calibration_yuan_json": json.dumps(
                            [
                                {
                                    "count": c["count"],
                                    "predicted_mean": c["predicted_mean"] * 100,
                                    "target_mean": c["target_mean"] * 100,
                                }
                                for c in s["calibration"]
                            ]
                        ),
                    }
                )
    with (root / "epoch-curves.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    fairness = {}
    for lr in ["3e-04", "1e-04", "3e-05"]:
        a, b = [result["fits"][f"lr{lr}-p{p}"] for p in [4, 8]]
        ma, mb = [load_model(root / f"lr{lr}-p{p}" / "best-dev.pt") for p in [4, 8]]
        same = parameter_hash(ma) == parameter_hash(mb)
        if not same or a["initial_parameter_sha256"] != b["initial_parameter_sha256"]:
            raise RuntimeError("patience comparison not identical at selected checkpoint")
        fairness[lr] = {
            "same_selected_parameters": same,
            "same_prefix_batch_order": all(
                x["batch_order_sha256"] == y["batch_order_sha256"]
                for x, y in zip(a["epochs"], b["epochs"])
            ),
        }
    data = Path("data/generated/mortal-v2/data")
    train, _, test = game_split(load_rows(data, json.loads((data / "manifest.json").read_text())))
    with gzip.open(SOURCE / "screening/decisions-faithful-mc-raw.jsonl.gz", "rt") as f:
        records = [json.loads(line) for line in f]
    templates = template_coverage(train, records)
    group = [r for r in test if str(r["family"]) == "hu/pass"]
    model = load_model(SOURCE / "faithful-mc/best-dev.pt").eval()
    with torch.inference_mode():
        q = model.forward_q(
            *pack([r["features"] for r in group], [r["actions"] for r in group], "cpu")
        )[0]
    p = q.gather(1, torch.tensor([int(r["chosen"]) for r in group])[:, None]).squeeze(-1).numpy()
    t = np.array([float(r["terminal_cash"]) for r in group])
    games = [int(r["game_id"]) for r in group]
    hu = np.array([chosen_kind(r) == "hu" for r in group])
    hudiag = {
        "mixed_correlation_ci95": clustered_correlation_ci(p, t, games),
        "hu_only_correlation_ci95": clustered_correlation_ci(p[hu], t[hu], np.asarray(games)[hu]),
        "pass_rows": [
            {
                "seed": int(r["seed"]),
                "seat": int(r["seat"]),
                "step": int(r["history_step"]),
                "cash": float(target) * 100,
                "q_cash": float(prediction) * 100,
            }
            for r, prediction, target in zip(group, p, t)
            if chosen_kind(r) == "pass"
        ],
    }
    outcomes = json.loads((SOURCE / "screening/screening-outcomes.json").read_text())
    guard = paired_summary(outcomes["faithful-mc-guarded"], outcomes["faithful-mc-raw"], 1414)
    atomic_json(
        root / "review.json",
        {
            "turning_points": comparisons,
            "patience_fairness": fairness,
            "template_coverage": templates,
            "hu_test_leverage": hudiag,
            "old_guard_minus_raw_cash": guard,
            "test_not_for_selection": True,
            "no_new_games": True,
        },
    )


if __name__ == "__main__":
    run()
