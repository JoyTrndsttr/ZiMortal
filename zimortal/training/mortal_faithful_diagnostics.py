"""Finite LR/patience diagnostics; test is observed but never selects checkpoints."""

import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from zimortal.model.encoding import KINDS
from zimortal.model.mortal_faithful import MortalFaithfulQNet

from .active import atomic_json
from .mortal_data import load_rows
from .mortal_diagnostics import distribution, metrics
from .mortal_offline import fit
from .mortal_v2 import game_split
from .online_rl import digest, pack, source_hashes
from .runtime import load_model

SOURCE = Path("data/generated/mortal-faithful-v1")


def cash_summary(rows):
    cash = np.array([round(float(r["terminal_cash"]) * 100, 4) for r in rows])
    counts = Counter(str(v) for v in cash)
    bins = {
        "negative": int((cash < 0).sum()),
        "zero": int((cash == 0).sum()),
        "positive_to50": int(((cash > 0) & (cash <= 50)).sum()),
        "positive50to100": int(((cash > 50) & (cash <= 100)).sum()),
        "positive100to200": int(((cash > 100) & (cash <= 200)).sum()),
        "positive_over200": int((cash > 200).sum()),
    }
    return {
        "decisions": len(rows),
        "games": len({r["game_id"] for r in rows}),
        "cash_yuan": distribution(cash),
        "cash_levels": dict(sorted(counts.items())),
        "cash_bins": bins,
        "chosen_behavior_probability": distribution(
            [r["behavior_probabilities"][int(r["chosen"])] for r in rows]
        ),
    }


def chosen_kind(row):
    return KINDS[int(row["actions"][int(row["chosen"]), : len(KINDS)].argmax())].value


def coverage(rows):
    return {
        family: {
            **cash_summary(group),
            "chosen_kind": {
                kind: cash_summary([r for r in group if chosen_kind(r) == kind])
                for kind in sorted({chosen_kind(r) for r in group})
            },
        }
        for family in sorted({str(r["family"]) for r in rows})
        for group in [[r for r in rows if str(r["family"]) == family]]
    }


def raw_diagnostics(path):
    with gzip.open(path, "rt") as file:
        rows = [json.loads(line) for line in file]
    result = {}
    for family in ["all", "discard", "chi/pass", "peng/pass", "hu/pass"]:
        group = rows if family == "all" else [r for r in rows if r["family"] == family]
        changed = [r for r in group if r["chosen"] != r["parent"]]
        pairs = []
        for r in group:
            p, q = np.array(r["probabilities"]), np.array(r["q"])
            keep = p > 0.1
            keep[r["parent"]] = True
            guard = int(np.where(keep, q, -np.inf).argmax())
            pairs.append(
                {
                    "raw": r["chosen"],
                    "guard": guard,
                    "raw_kind": r["actions"][r["chosen"]]["kind"],
                    "guard_kind": r["actions"][guard]["kind"],
                    "raw_p": p[r["chosen"]],
                    "guard_p": p[guard],
                    "raw_filtered": not bool(keep[r["chosen"]]),
                    "margin_cash": (np.sort(q)[-1] - np.sort(q)[-2]) * 100,
                    "changed": r["chosen"] != r["parent"],
                }
            )
        changed_pairs = [p for p in pairs if p["changed"]]
        result[family] = {
            "decisions": len(group),
            "overrides": len(changed),
            "override_probability": distribution([r["behavior_probability"] for r in changed]),
            "override_parent_probability": distribution([r["parent_probability"] for r in changed]),
            "override_p_le_01_fraction": float(np.mean([p["raw_p"] <= 0.1 for p in changed_pairs]))
            if changed_pairs
            else None,
            "raw_guard_different": sum(p["raw"] != p["guard"] for p in pairs),
            "raw_filtered": sum(p["raw_filtered"] for p in pairs),
            "override_margin_yuan": distribution([p["margin_cash"] for p in changed_pairs]),
            "low_support_override_margin_yuan": distribution(
                [p["margin_cash"] for p in changed_pairs if p["raw_p"] <= 0.1]
            ),
            "high_support_override_margin_yuan": distribution(
                [p["margin_cash"] for p in changed_pairs if p["raw_p"] > 0.1]
            ),
            "raw_to_guard_kinds": dict(
                Counter(
                    p["raw_kind"] + "->" + p["guard_kind"] for p in pairs if p["raw"] != p["guard"]
                )
            ),
            "large_margin_overrides": [
                {
                    "threshold_yuan": cut,
                    "count": int(sum(p["margin_cash"] >= cut for p in changed_pairs)),
                    "low_support_count": int(
                        sum(p["margin_cash"] >= cut and p["raw_p"] <= 0.1 for p in changed_pairs)
                    ),
                }
                for cut in [5, 15, 30]
            ],
        }
    return result


def hu_predictions(model, rows):
    group = [r for r in rows if str(r["family"]) == "hu/pass"]
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(group), 256):
            batch = group[start : start + 256]
            q = model.forward_q(
                *pack([r["features"] for r in batch], [r["actions"] for r in batch], "cuda")
            )[0]
            predictions.extend(
                q.gather(1, torch.tensor([int(r["chosen"]) for r in batch], device="cuda")[:, None])
                .squeeze(-1)
                .cpu()
                .tolist()
            )
    targets = np.array([float(r["terminal_cash"]) for r in group])
    prediction = np.array(predictions)
    kinds = np.array([chosen_kind(r) for r in group])
    result = {}
    for kind in ["all", "hu", "pass"]:
        mask = np.ones(len(group), bool) if kind == "all" else kinds == kind
        p, t = prediction[mask], targets[mask]
        if not len(p):
            result[kind] = {
                "count": 0,
                "mae": None,
                "rmse": None,
                "correlation": None,
                "q_std": None,
                "target_std": None,
            }
            continue
        result[kind] = {
            **metrics(p, t),
            "q_std": float(p.std()) if len(p) else None,
            "target_std": float(t.std()) if len(t) else None,
        }
        if len(p) > 1 and t.std() > 0:
            # Leave-one-game-out corr exposes leverage; several decisions may share a game.
            games = np.array([int(r["game_id"]) for r in group])[mask]
            correlations = []
            for game in np.unique(games):
                keep = games != game
                corr = metrics(p[keep], t[keep])["correlation"]
                if corr is not None:
                    correlations.append(corr)
            result[kind]["leave_one_game_out_correlation"] = distribution(correlations)
    return result


def run():
    torch.set_num_threads(2)
    torch.manual_seed(1414)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    root = Path("data/generated/mortal-faithful-diagnostics-v1")
    root.mkdir(parents=True, exist_ok=False)
    data = Path("data/generated/mortal-v2/data")
    manifest = json.loads((data / "manifest.json").read_text())
    original = json.loads((SOURCE / "config.json").read_text())
    for path, sha in original["inputs_sha256"].items():
        if digest(path) != sha:
            raise RuntimeError("original frozen inputs changed")
    train, dev, test = game_split(load_rows(data, manifest))
    checks = {
        str(data / "manifest.json"): digest(data / "manifest.json"),
        str(SOURCE / "faithful-mc/best-dev.pt"): digest(SOURCE / "faithful-mc/best-dev.pt"),
    }
    config = {
        "source_hashes": source_hashes(),
        "inputs_sha256": checks,
        "learning_rates": [3e-4, 1e-4, 3e-5],
        "patience": [4, 8],
        "epochs": 20,
        "test_diagnostic_only": True,
        "selection": "dev RMSE only",
        "alpha": 0.1,
        "support": 0.1,
        "target": "terminal MC cash/100",
        "loss": "0.5 MSE+CQL",
        "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
    }
    atomic_json(root / "config.json", config)
    atomic_json(
        root / "coverage.json",
        {n: coverage(rows) for n, rows in [("train", train), ("dev", dev), ("test", test)]},
    )
    atomic_json(
        root / "raw-screen-diagnostics.json",
        raw_diagnostics(SOURCE / "screening/decisions-faithful-mc-raw.jsonl.gz"),
    )
    original_model = load_model(SOURCE / "faithful-mc/best-dev.pt", "cuda")
    atomic_json(
        root / "original-hu-diagnostics.json",
        {
            n: hu_predictions(original_model, rows)
            for n, rows in [("train", train), ("dev", dev), ("test", test)]
        },
    )
    torch.manual_seed(1414)
    initial = MortalFaithfulQNet().cuda()
    results = {}
    for lr in config["learning_rates"]:
        for patience in config["patience"]:
            name = f"lr{lr:.0e}-p{patience}"
            results[name] = fit(
                initial,
                train,
                dev,
                test,
                root / name,
                device="cuda",
                alpha=0.1,
                n_step=0,
                epochs=20,
                patience=patience,
                regression_loss="mse",
                learning_rate=lr,
                epoch_diagnostics=True,
            )
    for path, sha in {**config["inputs_sha256"], **config["source_hashes"]}.items():
        if digest(path) != sha:
            raise RuntimeError("source/input changed during finite diagnostic")
    atomic_json(
        root / "result.json",
        {
            "status": "complete_diagnostic_stop",
            "fits": results,
            "test_used_for_selection": False,
            "online_started": False,
        },
    )


if __name__ == "__main__":
    run()
