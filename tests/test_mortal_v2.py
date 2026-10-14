import numpy as np
import pytest

from zimortal.engine import Action, ActionType
from zimortal.training.mortal_data import family_exploration
from zimortal.training.mortal_diagnostics import diagnostics, select_gate
from zimortal.training.mortal_v2 import game_split


def test_family_temperature_hu_pass_and_reproducible_legal_sampling():
    config = {
        "discard": 1.5,
        "chi/pass": 1.5,
        "peng/pass": 2.5,
        "hu/pass": 1.0,
        "hu_pass_probability": 0.02,
    }
    actions = [Action(ActionType.HU, 0), Action(ActionType.PASS, 0)]
    p = family_exploration([100, -100], actions, config)
    assert p.tolist() == pytest.approx([0.98, 0.02])
    a = np.random.default_rng(12).choice(2, size=1000, p=p)
    b = np.random.default_rng(12).choice(2, size=1000, p=p)
    assert np.array_equal(a, b) and set(a) == {0, 1}
    actions[0] = Action(ActionType.PENG, 0)
    assert family_exploration([5, 0], actions, config)[1] > 0.1
    config["hu_pass_probability"] = 0.1
    actions[0] = Action(ActionType.HU, 0)
    with pytest.raises(ValueError):
        family_exploration([5, 0], actions, config)


def test_margin_gate_and_full_game_split():
    rows = [{"game_id": i, "family": "discard", "parent_greedy": 0} for i in range(10)]
    train, dev, test = game_split(rows * 2)
    assert len(train) == 16 and len(dev) == len(test) == 2
    q = [np.array([0.0, 0.05]) for _ in rows]
    report = diagnostics(rows, np.arange(10.0), np.arange(10.0), q)
    assert report["q_margin"]["all"]["p50"] == pytest.approx(0.05)
    assert report["gate_rates"]["0.05"] == 0
    assert select_gate(report) == 0.05
    assert all(b["predicted_mean"] == b["target_mean"] for b in report["calibration"])


def test_early_stopping_uses_development_not_training(monkeypatch, tmp_path):
    import zimortal.training.mortal_offline as module
    from zimortal.model.encoding import ACTION_DIM
    from zimortal.model.mortal_q import MortalQNet

    values = iter([2.0, 2.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])

    def regression(*args, **kwargs):
        return {"rmse": next(values)}

    monkeypatch.setattr(module, "regression", regression)
    row = {
        "features": np.zeros((55, 20), np.float32),
        "actions": np.zeros((2, ACTION_DIM), np.float32),
        "chosen": 0,
        "terminal_cash": 0.0,
        "steps_to_terminal": 0,
    }
    result = module.fit(
        MortalQNet(width=8, latent=32, blocks=1),
        [row],
        [row],
        [row],
        tmp_path / "fit",
        epochs=20,
        patience=2,
    )
    assert result["best_epoch"] == 1 and result["epochs_run"] == 3
    assert result["early_stopped"]
