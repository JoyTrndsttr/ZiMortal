"""Audit saved aggregates; separate same-game revisits from independent outcomes."""

import argparse
import json
from pathlib import Path

import numpy as np

from zimortal.training.active import atomic_json
from zimortal.training.mortal_credit_diagnostics import key_pairs, summary
from zimortal.training.online_rl import digest


def independent_coverage(key, action, game):
    _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    ng = np.bincount(np.unique(key_pairs(inv, game))["a"], minlength=len(counts))
    na = np.bincount(np.unique(key_pairs(inv, action))["a"], minlength=len(counts))
    qualified = (ng >= 2) & (na >= 2)
    histogram = {}
    for lo, hi in ((1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 20), (21, 2**63 - 1)):
        histogram[f"{lo}..{hi}" if hi < 100 else "21+"] = int(
            np.sum((counts >= lo) & (counts <= hi))
        )
    return {
        "rows": len(key),
        "states": len(counts),
        "visit_count_distribution": summary(counts),
        "visit_histogram_states": histogram,
        "max_visits": int(counts.max()),
        "cross_game_multi_action_states": int(qualified.sum()),
        "cross_game_multi_action_state_fraction": float(qualified.mean()),
        "cross_game_multi_action_row_fraction": float(qualified[inv].mean()),
        "distinct_executed_actions_distribution": summary(na),
        "independent_games_distribution": summary(ng),
    }


def run(root):
    config = json.loads((root / "config.json").read_text())
    result = json.loads((root / "result.json").read_text())
    assert (
        result["status"] == "complete_stop"
        and result["new_games"] == result["optimizer_steps"] == 0
    )
    if "diagnostic_source_correction" in config:
        fix = config["diagnostic_source_correction"]
        old = Path(fix["original_source"]).read_text()
        assert digest(fix["original_source"]) == fix["original_sha256"]
        expected_old = 'np.testing.assert_allclose(q, r["recorded_q"], atol=1e-5, rtol=1e-5)'
        assert fix["replacement_old"] == expected_old
        expected_new = expected_old.replace("atol=1e-5", "atol=5e-4")
        assert fix["replacement_new"] == expected_new
        assert old.count(expected_old) == 1
        assert old.replace(expected_old, fix["replacement_new"]) == Path(fix["file"]).read_text()
        original_config = json.loads((root / "config-before-tolerance-fix.json").read_text())
        original_config["source_sha256"][fix["file"]] = config["source_sha256"][fix["file"]]
        assert original_config == {
            k: v for k, v in config.items() if k != "diagnostic_source_correction"
        }
        assert digest(root / "recover-gradients.py") == result["recovery_script_sha256"]
        for p, sha in json.loads((root / "recovery-aggregate-sha256.json").read_text()).items():
            assert digest(p) == sha, p
    frozen = json.loads((root / "input-sha256.json").read_text()) | config["source_sha256"]
    for p, sha in frozen.items():
        assert digest(p) == sha, p
    reports = {}
    for stream in config["streams"]:
        with np.load(root / f"{stream}-statistics.npz", allow_pickle=False) as npz:
            data = npz["rows"]
        stored = result["streams"][stream]
        assert len(data) == stored["rows"]
        reports[stream] = {}
        for f in ("all", "discard", "chi/pass", "peng/pass", "hu/pass"):
            ix = (
                np.ones(len(data), bool)
                if f == "all"
                else data["family"] == ("discard", "chi/pass", "peng/pass", "hu/pass").index(f)
            )
            selected = data[ix]
            if not len(selected):
                continue
            reports[stream][f] = {}
            for field in ("exact", "encoded", "fine", "broad", "compatible"):
                stats = independent_coverage(selected[field], selected["action"], selected["game"])
                assert stats["states"] == stored["families"][f]["coverage"][field]["states"]
                reports[stream][f][field] = stats
            pair = key_pairs(selected["compatible"], selected["generation"])
            reports[stream][f]["compatible_same_generation"] = independent_coverage(
                pair, selected["action"], selected["game"]
            )
        atomic_json(root / "independent-coverage-review.json", reports)
    for grad in result["gradients"].values():
        for mode in ("eval", "train_isolated_BN"):
            assert grad[mode]["state_hash_unchanged"]
            assert grad[mode]["direct_derivative_max_abs_error"] < 1e-5
    atomic_json(
        root / "audit-result.json",
        {
            "input_hashes_verified": True,
            "aggregation_verified": True,
            "new_games": 0,
            "optimizer_steps": 0,
            "independent_outcome_definition": "at least two distinct actions AND at least two distinct recorded games",
            "source_sha256": digest("scripts/audit-mortal-credit-assignment.py"),
        },
    )
    print("Saved statistics, original input hashes and cross-game action coverage verified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory", type=Path, default=Path("data/generated/mortal-credit-diagnostics-v1")
    )
    run(parser.parse_args().directory)
