import numpy as np
import pytest

from zimortal.training.mortal_data_volume import ShuffledBatches, train_fixed


def test_fixed_batches_cross_cycle_without_losing_examples():
    batches = ShuffledBatches(5, seed=8)
    a = batches.next(3)
    b = batches.next(3)
    assert len(a) == len(b) == 3
    assert sorted(np.concatenate([a, b[:2]])) == list(range(5))
    replica = ShuffledBatches(5, seed=8)
    assert np.array_equal(replica.next(6), np.concatenate([a, b]))
    assert batches.position == 1


def test_invalid_fixed_budget_rejected_before_training(tmp_path):
    with pytest.raises(ValueError):
        train_fixed(None, [], [], tmp_path / "bad", updates=521, interval=260)


def test_dev_selects_same_update_for_both_models():
    from zimortal.training.mortal_data_volume import select_common_update

    fits = {
        "A": {
            "logs": [
                {"update": 260, "development": {"rmse": 1}},
                {"update": 520, "development": {"rmse": 2}},
            ]
        },
        "B": {
            "logs": [
                {"update": 260, "development": {"rmse": 3}},
                {"update": 520, "development": {"rmse": 1}},
            ]
        },
    }
    assert select_common_update(fits) == 520
    fits["B"]["logs"][0]["update"] = 100
    with pytest.raises(ValueError):
        select_common_update(fits)
