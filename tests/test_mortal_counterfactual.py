import numpy as np
import pytest

from zimortal.engine import RuleEngine
from zimortal.model.encoding import encode_observation
from zimortal.model.network import PolicyValueNet
from zimortal.training.mortal_counterfactual import (
    Frequencies,
    paired_rollouts,
    paired_statistics,
    particle,
    ranks,
    sample_roots,
)


def decision_root(seed=118):
    engine = RuleEngine()
    state = engine.new_game(seed, 0)
    while len(engine.legal_actions(state)) == 1:
        state = engine.step(state, engine.legal_actions(state)[0])
    return state, engine.legal_actions(state)


def test_common_particle_preserves_hands_public_features_and_root():
    root, actions = decision_root()
    original = root.serialize()
    a, b = particle(root, 21), particle(root, 21)
    assert a.serialize() == b.serialize() and root.serialize() == original
    assert sorted(a.deck) == sorted(root.deck)
    assert [p.hand for p in a.players] == [p.hand for p in root.players]
    engine = RuleEngine()
    np.testing.assert_array_equal(
        encode_observation(engine.observation(root, actions[0].player), "huxi"),
        encode_observation(engine.observation(a, actions[0].player), "huxi"),
    )


def test_paired_rollout_reproducible_and_visible_frozen_continuation():
    root, actions = decision_root()
    original = root.serialize()
    model = PolicyValueNet(width=8, feature_version="huxi").eval().requires_grad_(False)
    a, e = paired_rollouts(root, actions[0], actions[1], model, count=2, seed=17)
    b, f = paired_rollouts(root, actions[0], actions[1], model, count=2, seed=17)
    np.testing.assert_array_equal(a, b)
    assert e == f and root.serialize() == original
    assert all(r["parent_terminal"] and r["selected_terminal"] for r in e)
    assert all(p.grad is None for p in model.parameters())
    with pytest.raises(ValueError):
        paired_rollouts(root, actions[0], actions[0], model)


def test_pair_difference_se_uses_covariance_and_zero_is_not_certified_tie():
    values = np.array([[0, 7], [100, 107], [-100, -93]], float)
    stats = paired_statistics(values)
    assert stats["paired_delta"] == 7 and stats["paired_standard_error"] == 0
    assert stats["paired_bootstrap_95"] == [7, 7]
    assert stats["resolved_sign"] == "positive"
    zero = paired_statistics(np.zeros((32, 2)))
    assert zero["sign"] == "sample_zero" and zero["resolved_sign"] == "unresolved"


def test_quartile_sampling_no_outcome_selection_and_correct_weights():
    population = [
        {"family": "discard", "pair_q_gap": i, "seed": i, "seat": 0, "step": 1} for i in range(12)
    ]
    samples = sample_roots(population, 8, 21)
    assert samples == sample_roots(population, 8, 21)
    assert len(samples) == len({r["seed"] for r in samples}) == 8
    assert {r["sampling_stratum"] for r in samples} == {0, 1, 2, 3}
    assert all(r["sampling_weight"] == 1.5 for r in samples)
    assert not [r for r in samples if r["family"] == "hu/pass"]


def test_actual_frequency_separates_exact_state_and_global_action():
    x = np.zeros((55, 20), np.float32)
    actions = np.array([[1, 0], [0, 1]], np.float32)
    index = Frequencies([{"features": x, "actions": actions, "chosen": 0}])
    seen = index.lookup(x, actions, 0)
    assert (
        seen["chosen_training_rows"]
        == seen["eligible_training_rows"]
        == seen["exact_input_chosen_rows"]
        == 1
    )
    unseen = index.lookup(x + 1, actions, 0)
    assert unseen["chosen_training_rows"] == 1 and unseen["exact_input_training_rows"] == 0
    other = index.lookup(x, actions, 1)
    assert other["chosen_training_rows"] == 0 and other["eligible_training_rows"] == 1


def test_rank_ties_and_missing_correlation():
    from zimortal.training.mortal_counterfactual import correlation, ranking_summary

    np.testing.assert_array_equal(ranks([3, 1, 1]), [3, 1.5, 1.5])
    assert correlation([1, 1], [2, 3]) is None
    assert ranking_summary([])["roots"] == 0


def test_independent_confirmation_distinguishes_point_and_interval():
    from zimortal.training.mortal_counterfactual_confirm import comparison

    first = {"sign": "negative", "resolved_sign": "negative"}
    assert comparison(first, {"sign": "negative", "resolved_sign": "unresolved"}) == {
        "same_point_sign": True,
        "same_ci_sign": False,
        "opposite_ci_sign": False,
    }
    assert comparison(first, {"sign": "positive", "resolved_sign": "positive"})["opposite_ci_sign"]
    with pytest.raises(ValueError):
        comparison({"resolved_sign": "unresolved"}, first)


def test_family_frequency_does_not_mix_pass_contexts():
    x = np.zeros((55, 20), np.float32)
    actions = np.array([[1, 0], [0, 1]], np.float32)
    index = Frequencies(
        [
            {"features": x, "actions": actions, "chosen": 0, "family": "chi/pass"},
            {"features": x + 1, "actions": actions, "chosen": 1, "family": "peng/pass"},
        ]
    )
    global_count = index.lookup(x, actions, 0)["chosen_training_rows"]
    chi = index.lookup(x, actions, 0, "chi/pass")
    peng = index.lookup(x, actions, 0, "peng/pass")
    assert global_count == chi["family_chosen_training_rows"] == 1
    assert peng["family_chosen_training_rows"] == 0 and peng["family_eligible_training_rows"] == 1
