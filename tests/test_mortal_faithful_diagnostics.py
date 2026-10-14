import gzip
import json

import numpy as np

from zimortal.model.encoding import ACTION_DIM, KINDS
from zimortal.training.mortal_faithful_diagnostics import coverage, raw_diagnostics


def test_raw_same_state_guard_and_json_serialization(tmp_path):
    path = tmp_path / "decisions.gz"
    record = {
        "chosen": 1,
        "parent": 0,
        "family": "hu/pass",
        "probabilities": [0.98, 0.02],
        "q": [0.0, 0.5],
        "behavior_probability": 0.02,
        "parent_probability": 0.98,
        "actions": [{"kind": "hu"}, {"kind": "pass"}],
    }
    with gzip.open(path, "wt") as f:
        f.write(json.dumps(record) + "\n")
    result = raw_diagnostics(path)
    json.dumps(result, allow_nan=False)
    assert result["hu/pass"]["raw_to_guard_kinds"] == {"pass->hu": 1}
    assert result["all"]["large_margin_overrides"][2]["low_support_count"] == 1


def test_coverage_cash_units_and_chosen_categories():
    action = np.zeros((1, len(KINDS)), np.float32)
    action[0, next(i for i, k in enumerate(KINDS) if k.value == "hu")] = 1
    row = {
        "actions": action,
        "chosen": 0,
        "family": "hu/pass",
        "terminal_cash": 1.5,
        "behavior_probabilities": [0.98],
        "game_id": 3,
    }
    result = coverage([row])["hu/pass"]
    assert result["cash_yuan"]["mean"] == 150
    assert result["chosen_kind"]["hu"]["decisions"] == 1
    assert result["cash_bins"]["positive100to200"] == 1


def test_template_counts_do_not_confuse_low_probability_with_globally_unseen():
    from zimortal.engine import Action, ActionType
    from zimortal.model.encoding import encode_action
    from zimortal.training.mortal_action_coverage import template_coverage

    action = Action(ActionType.DISCARD, 0, tile=0)
    row = {"family": "discard", "actions": encode_action(action, 0)[None], "chosen": 0}
    record = {
        "family": "discard",
        "chosen": 1,
        "parent": 0,
        "margin": 0.01,
        "behavior_probability": 0.001,
        "actions": [
            {"kind": "discard", "player": 0, "tile": 1},
            {"kind": "discard", "player": 0, "tile": 0},
        ],
    }
    result = template_coverage([row], [record])["discard"]
    assert result["zero_chosen_template_count"] == 0
    assert result["low_support_nonzero_template_count"] == 1


def test_epoch_test_diagnostics_cannot_choose_checkpoint(monkeypatch, tmp_path):
    from zimortal.model.mortal_q import MortalQNet
    from zimortal.training import mortal_offline

    row = {
        "features": np.zeros((55, 20), np.float32),
        "actions": np.zeros((2, ACTION_DIM), np.float32),
        "chosen": 0,
        "terminal_cash": 0,
        "steps_to_terminal": 0,
    }
    train, dev, test = [row], [row], [row]
    counts = {"dev": 0, "test": 0}

    def regression(model, rows, **kwargs):
        if rows is dev:
            counts["dev"] += 1
            return {"rmse": 2.0 if counts["dev"] == 3 else 1.0}
        if rows is test:
            counts["test"] += 1
            return {"rmse": 1.0 / counts["test"]}
        return {"rmse": 1.0}

    monkeypatch.setattr(mortal_offline, "regression", regression)
    result = mortal_offline.fit(
        MortalQNet(width=8, latent=16, blocks=1),
        train,
        dev,
        test,
        tmp_path / "diagnostic",
        epochs=3,
        patience=1,
        epoch_diagnostics=True,
        learning_rate=1e-4,
    )
    assert result["best_epoch"] == 1
    assert result["epochs_run"] == 2
    assert (
        result["epochs"][1]["test_diagnostic_only"]["rmse"]
        < result["epochs"][0]["test_diagnostic_only"]["rmse"]
    )
