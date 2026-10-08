"""No privileged hidden world enters the rollout teacher."""

import random

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
from zimortal.belief.sampling import SamplingFailure, sample_world
from zimortal.engine import RuleEngine
from zimortal.training.rollout import soft_target, teacher


def test_hidden_particles_replay_every_public_action():
    engine = RuleEngine()
    state = engine.new_game(118)
    for _ in range(10):
        actions = engine.legal_actions(state)
        state = engine.step(state, random.Random(118).choice(actions))
    actions = engine.legal_actions(state)
    obs = engine.observation(state, actions[0].player)
    world, attempts = sample_world(obs, 77, max_attempts=10000)
    world.validate()
    assert attempts >= 1
    assert engine.observation(world, obs.player) == obs
    again, n = sample_world(obs, 77, max_attempts=10000)
    assert world.serialize() == again.serialize() and n == attempts


def test_no_inconsistent_sampling_fallback():
    engine = RuleEngine()
    state = engine.new_game(118)
    obs = engine.observation(state, state.turn)
    with pytest.raises(SamplingFailure):
        sample_world(obs, 0, max_attempts=0)


def test_cash_soft_targets_preserve_order_and_units():
    p = soft_target([200, 0, -100], 10)
    assert p[0] > p[1] > p[2] and sum(p) == pytest.approx(1)
    assert soft_target([210, 10, -90], 10) == pytest.approx(p)
    with pytest.raises(ValueError):
        soft_target([1, 2], 0)


def test_rollout_targets_use_all_candidates_and_real_payments():
    engine = RuleEngine()
    state = engine.new_game(118)
    state = engine.step(state, engine.legal_actions(state)[0])
    actions = engine.legal_actions(state)
    obs = engine.observation(state, actions[0].player)
    result = teacher(obs, rollouts=2, seed=144)
    assert len(result.q_cash) == len(actions)
    assert len(result.policy) == len(actions) and sum(result.policy) == pytest.approx(1)
    assert result.rollouts_per_action == 2
    assert all(np.isfinite(result.q_cash))


def test_planning_heads_mask_impossible_deck_counts(tmp_path):
    import torch

    from zimortal.model.encoding import encode_action, encode_observation
    from zimortal.model.network import PolicyValueNet
    from zimortal.training.runtime import load_model, save_model

    engine = RuleEngine()
    state = engine.new_game(118)
    obs = engine.observation(state, state.turn)
    model = PolicyValueNet(feature_version="huxi", auxiliary_version="planning")
    x = torch.tensor(encode_observation(obs, "huxi")[None])
    a = torch.tensor(np.stack([encode_action(t, obs.player) for t in obs.legal_actions])[None])
    mask = torch.ones(a.shape[:2], dtype=torch.bool)
    outputs, belief, distance = model.forward_planning(x, a, mask)
    upper = (4 - (x[:, 0] + x[:, 1]) * 4).round()
    for tile in range(20):
        assert torch.all(belief[0, tile, int(upper[0, tile]) + 1 :] == -1e9)
    from zimortal.training.planning import uniform_belief

    torch.testing.assert_close(
        belief.softmax(-1),
        torch.tensor(uniform_belief(x.numpy()), dtype=belief.dtype),
        rtol=1e-4,
        atol=1e-5,
    )
    assert belief.shape == (1, 20, 5) and distance.shape == (1, 16)
    assert outputs[0].shape == mask.shape
    path = tmp_path / "planning.pt"
    save_model(model, path, value_target="net payments/100")
    loaded = load_model(path)
    assert loaded.auxiliary_version == "planning"
    torch.testing.assert_close(loaded.forward_planning(x, a, mask)[1], belief)
