import random
from unittest.mock import patch

import pytest

from zimortal.engine import RuleEngine
from zimortal.training.rollout_engine import RolloutEngine, cached_chi, cached_evaluate_hand


@pytest.mark.parametrize("seed", [1, 46, 118, 322, 44501])
def test_cached_rollout_preserves_actions_observations_and_every_transition(seed):
    original, cached = RuleEngine(), RolloutEngine()
    state = original.new_game(seed, dealer=seed % 3)
    rng = random.Random(seed)
    for _ in range(1000):
        if state.terminal:
            break
        actions = original.legal_actions(state)
        expected_observations = [original.observation(state, seat) for seat in range(3)]
        action = rng.choice(actions)
        expected = original.step(state, action)
        with (
            patch("zimortal.engine.game.evaluate_hand", cached_evaluate_hand),
            patch("zimortal.engine.game.enumerate_chi_with_required_bi", cached_chi),
        ):
            assert cached.legal_actions(state) == actions
            for seat in range(3):
                assert cached.observation(state, seat) == expected_observations[seat]
            actual = cached.step(state, action)
        actual.validate()
        assert actual.serialize() == expected.serialize()
        state = actual
    assert state.terminal


def test_cache_reuses_only_same_state_and_keeps_step_legality_check():
    class CountingEngine(RolloutEngine):
        calls = 0

        def _legal(self, state):
            self.calls += 1
            return super()._legal(state)

    engine = CountingEngine()
    state = engine.new_game(118)
    actions = engine.legal_actions(state)
    engine.observation(state, actions[0].player)
    assert engine.calls == 1
    next_state = engine.step(state, actions[0])
    assert engine.calls == 1
    engine.legal_actions(next_state)
    assert engine.calls == 2
    with pytest.raises(ValueError, match="illegal action"):
        engine.step(next_state, actions[0])
