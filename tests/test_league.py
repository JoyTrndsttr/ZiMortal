import argparse
import json

import pytest

from zimortal.training import league
from zimortal.training.online_rl import aggregate_summary


def evidence(lower=1, parent=1, worst=0):
    return {
        "valid": True,
        "aggregate": {"paired_seed_bootstrap_interval": [lower, 5]},
        "opponents": {
            "0": {"paired_seed_bootstrap_interval": [parent, 5], "mean_cash_delta": 2},
            "1": {"mean_cash_delta": worst},
        },
    }


def test_promotion_requires_aggregate_parent_and_regression_checks():
    assert league.promotion(evidence(), 3)["accepted"]
    assert not league.promotion(evidence(lower=-1), 3)["accepted"]
    assert not league.promotion(evidence(parent=0), 3)["accepted"]
    assert not league.promotion(evidence(worst=-4), 3)["accepted"]
    assert not league.promotion({"valid": False}, 3)["accepted"]


def test_aggregate_keeps_opponents_and_seats_in_same_seed_cluster():
    reference = [
        {"seed": s, "seat": seat, "opponent": o, "cash": 0}
        for s in (100, 101)
        for seat in range(3)
        for o in range(2)
    ]
    candidate = [{**r, "cash": 6 if r["seed"] == 100 else -3} for r in reference]
    report = aggregate_summary(candidate, reference, 1, 0.0125)
    assert report["independent_seeds"] == 2
    assert report["mean_cash_delta"] == 1.5
    assert report["paired_seed_bootstrap_interval"] == [-3, 6]
    with pytest.raises(RuntimeError):
        aggregate_summary(candidate[:-1], reference, 1)


def test_continuation_keeps_optimizer_and_only_accepted_parent_changes(tmp_path, monkeypatch):
    initial, anchor, historical = (tmp_path / n for n in ("initial.pt", "anchor.pt", "history.pt"))
    for p in (initial, anchor, historical):
        p.write_bytes(p.name.encode())
    args = argparse.Namespace(
        initial=str(initial),
        anchor=str(anchor),
        historical=str(historical),
        initial_optimizer=None,
        opponents=[],
        directory=str(tmp_path / "data"),
        reports=str(tmp_path / "reports"),
        stages=3,
        generations=2,
        games=18,
        later_games=36,
        slots=2,
        lr=3e-5,
        seed=10,
        train_start=1000,
        dev_start=2000,
        screen_start=3000,
        seed_stride=100,
        dev_seeds=2,
        screen_seeds=2,
        final_start=4000,
        final_seeds=2,
        device="cpu",
        snapshot_pool=2,
        regression_tolerance=3,
        commit_reports=False,
    )
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "final-audit.json").write_text("{}")
    calls = []

    def simulate(settings):
        from pathlib import Path

        calls.append(settings)
        root, report = Path(settings.directory), Path(settings.reports)
        root.mkdir(parents=True)
        report.mkdir(parents=True)
        for name in ("best-dev.pt", "generation-002.pt", "last.pt"):
            (root / name).write_bytes((root.name + name).encode())
        (report / "generation-001.json").write_text(
            json.dumps(
                {"completed_games": settings.games, "decisions": 100, "excluded_rule_games": 0}
            )
        )
        (report / "holdout.json").write_text(
            json.dumps({"evaluation": evidence(lower=1 if len(calls) == 2 else -1)})
        )

    monkeypatch.setattr(league, "run", simulate)
    league.league(args)
    assert len(calls) == 3
    assert calls[1].base == str(initial)
    assert calls[1].initialize.endswith("stage-01/generation-002.pt")
    assert calls[1].optimizer_checkpoint.endswith("stage-01/last.pt")
    assert not calls[1].optimizer_best
    assert calls[2].base.endswith("stage-02/best-dev.pt")
    assert calls[2].initialize == calls[2].base and calls[2].optimizer_best
    assert calls[2].optimizer_checkpoint.endswith("stage-02/last.pt")
    assert str(initial) in calls[2].opponents
    assert len({c.train_start for c in calls}) == len({c.holdout_start for c in calls}) == 3
    league.league(args)
    assert len(calls) == 3  # Completed stages are resumed, not trained twice.
    done = json.loads((tmp_path / "data/league-progress.json").read_text())
    assert done["status"] == "complete" and len(done["promoted_models"]) == 1
