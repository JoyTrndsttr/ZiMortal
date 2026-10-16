import json
from functools import partial

import numpy as np
import pytest
import torch

from zimortal.model.mortal_q import MortalQNet
from zimortal.training.mortal_lr_horizon import (
    bn_change,
    buffer_snapshot,
    extreme_stop,
    horizon_protocol,
    optimizer_dynamics,
    q_dynamics,
    schedule,
    seed_specs,
)
from zimortal.training.mortal_online import learning_rate, make_optimizer, update


def test_schedule_is_only_timescale_change_and_forecast_is_historical(tmp_path):
    for g in range(1, 11):
        p = tmp_path / f"gen-{g:02d}"
        p.mkdir()
        (p / "training-result.json").write_text(json.dumps({"decisions": 41528 + (g <= 4)}))
    protocol = horizon_protocol(tmp_path)
    assert protocol["historical_rows"] == 415284
    assert protocol["expected_updates"] == 1630
    assert protocol["B_cosine_updates"] == 1609
    assert [schedule(s, 300) for s in range(2000)] == [learning_rate(s) for s in range(2000)]
    assert schedule(0, 1609) == pytest.approx(3e-6)
    assert schedule(20, 1609) == pytest.approx(3e-5)
    assert schedule(1629, 1609) == pytest.approx(3e-6)
    assert schedule(100000, 1609) == pytest.approx(3e-6)
    assert schedule(320, 1609) > learning_rate(320)
    assert all(schedule(s, 1609) >= schedule(s + 1, 1609) for s in range(20, 1800))


def test_new_seeds_and_fixed_holdout_are_disjoint_and_roles_match():
    seen = set()
    for g in range(1, 11):
        for evaluation in (False, True):
            specs = seed_specs(g, evaluation=evaluation)
            seeds = {s[0] for s in specs}
            assert not seen & seeds and min(seeds) >= 1270000000
            seen |= seeds
            assert len(specs) == (3000 if evaluation else 5000)
            if not evaluation:
                assert {(d, s) for _, d, s, _ in specs} == {
                    (d, s) for d in range(3) for s in range(3)
                }
    holdout = seed_specs(holdout=True)
    assert len(holdout) == 15000 and not seen & {s[0] for s in holdout}
    with pytest.raises(ValueError):
        seed_specs(1, holdout=True)


def test_q_drift_separates_common_from_policy_relevant_advantage():
    rows = [{"reference_probabilities": np.array([0.5, 0.5])}]
    before = [np.array([0.0, 1.0])]
    common = q_dynamics(before, [np.array([3.0, 4.0])], rows)
    assert common["common_square_fraction"] == pytest.approx(1)
    assert common["centered_advantage_rms"] == pytest.approx(0)
    assert common["raw_argmax_change_rate"] == common["anchored_argmax_change_rate"] == 0
    changed = q_dynamics(before, [np.array([2.0, -1.0])], rows)
    assert changed["common_square_fraction"] == pytest.approx(0)
    assert changed["centered_advantage_rms"] == pytest.approx(2)
    assert changed["raw_argmax_change_rate"] == changed["anchored_argmax_change_rate"] == 1
    with pytest.raises(ValueError):
        q_dynamics([], [], [])


def test_bn_buffers_are_not_parameters_and_schedule_keeps_adam_continuous():
    bn = torch.nn.BatchNorm1d(2)
    before = buffer_snapshot(bn)
    bn(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    drift = bn_change(before, bn)
    assert drift["running_mean"]["delta_l2"] > 0
    assert drift["num_batches_tracked"]["delta_l2"] == 1
    # Identical schedules must be bit-identical to the original update implementation.
    torch.manual_seed(12)
    a = MortalQNet(width=4, latent=8, blocks=1)
    b = MortalQNet(width=4, latent=8, blocks=1)
    b.load_state_dict(a.state_dict())
    rows = [
        {
            "features": np.zeros((55, 20), np.float32),
            "actions": np.zeros((2, 49), np.float32),
            "chosen": 0,
            "terminal_cash": 0.5,
            "steps_to_terminal": 0,
        }
        for _ in range(3)
    ]
    from zimortal.model.encoding import ACTION_DIM

    for r in rows:
        r["actions"] = np.zeros((2, ACTION_DIM), np.float32)
    oa, ob = make_optimizer(a), make_optimizer(b)
    ra, rb = np.random.default_rng(7), np.random.default_rng(7)
    ua, la = update(a, oa, rows, ra, 0, batch_size=2)
    ub, lb = update(
        b, ob, rows, rb, 0, batch_size=2, lr_schedule=partial(schedule, cosine_updates=300)
    )
    assert ua == ub and la == lb and ra.bit_generator.state == rb.bit_generator.state
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    ub, logs = update(
        b, ob, rows, rb, ub, batch_size=2, lr_schedule=partial(schedule, cosine_updates=1609)
    )
    d = optimizer_dynamics(logs, ob)
    assert d["adam_steps"] == [ub] and d["cumulative_updates"] == ub
    assert sum(l["batch"] for l in logs) == len(rows)


def test_safety_only_two_consecutive_extreme_generations():
    def r(a, b):
        return {
            "cash": {
                f"{arm}-anchor": {"seed_bootstrap_95": [-10, v]} for arm, v in [("A", a), ("B", b)]
            }
        }

    assert not extreme_stop([r(-6, 0)])
    assert not extreme_stop([r(-6, 0), r(0, -6)])
    assert not extreme_stop([r(-6, 0), r(-5, 0)])
    assert extreme_stop([r(-6, 0), r(-6, 0)])
