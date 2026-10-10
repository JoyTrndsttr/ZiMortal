import random
from dataclasses import replace

import numpy as np

from zimortal.engine import Meld, MeldType, RuleEngine
from zimortal.engine.observation import PublicPlayer
from zimortal.engine.scoring import settle
from zimortal.training.adaptive import empirical_bernstein
from zimortal.training.adaptive import payoff_bounds as original_bounds
from zimortal.training.payoff_envelope import group_huxi_upper, payoff_bounds


def test_bernstein_promotes_int16_support_before_range_multiplication():
    expected = empirical_bernstein(20, 0, 128, -8000, 10000, 0.001)
    with np.errstate(over="raise"):
        actual = empirical_bernstein(20, 0, np.int64(128), np.int16(-8000), np.int16(10000), 0.001)
    assert actual == expected
    assert actual[0] < 0 < actual[1]


def test_relaxed_group_capacity_conserves_size_and_distinct_quad_types():
    assert group_huxi_upper(6, 24, 0, 6, 0) == 72
    assert group_huxi_upper(6, 0, 24, 0, 6) == 54
    assert group_huxi_upper(6, 18, 0, 0, 0) == 36
    assert group_huxi_upper(6, 18, 0, 6, 0) == 36
    assert group_huxi_upper(6, 8, 16, 2, 4) == 60


def test_existing_wei_can_all_upgrade_without_subtracting_future_fourth_twice():
    engine = RuleEngine()
    state = engine.new_game(118)
    obs = engine.observation(state, 0)
    wei = tuple(Meld(MeldType.WEI, (t,) * 3) for t in range(12, 18))
    obs = replace(
        obs,
        hand=(18, 18),
        players=(PublicPlayer(2, wei), PublicPlayer(20, ()), PublicPlayer(20, ())),
    )
    groups = tuple(Meld(MeldType.TI, (t,) * 4) for t in range(12, 18)) + (
        Meld(MeldType.PAIR, (18, 18)),
    )
    assert payoff_bounds(obs)[1] >= settle(0, groups, ordinary_fan=1).payments[0]


def test_tighter_envelope_contains_real_future_settlements_at_all_observed_steps():
    engine = RuleEngine()
    tightened = wins = 0
    for seed in range(160):
        state = engine.new_game(seed, dealer=seed % 3)
        rng = random.Random(seed + 31000)
        envelopes = []
        for _ in range(1000):
            if state.terminal:
                break
            actions = engine.legal_actions(state)
            obs = engine.observation(state, actions[0].player)
            old, new = original_bounds(obs), payoff_bounds(obs)
            assert new[0] >= old[0] and new[1] <= old[1]
            tightened += new != old
            envelopes.append((obs.player, new))
            state = engine.step(state, rng.choice(actions))
        assert state.terminal
        state.validate()
        payments = state.settlement.payments if state.settlement else (0, 0, 0)
        wins += state.settlement is not None
        for player, (low, high) in envelopes:
            assert low <= payments[player] <= high
    assert tightened > 100
    assert wins > 0
