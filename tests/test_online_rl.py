import copy
import random

import numpy as np
import pytest
import torch

from zimortal.engine import RuleEngine
from zimortal.model.network import PolicyValueNet
from zimortal.training.online_rl import infer, paired_summary, play, ppo_update


def test_cash_selfplay_batched_collection_is_reproducible_and_updates_policy():
    torch.set_num_threads(1)
    torch.manual_seed(27)
    model = PolicyValueNet("mlp", width=8, feature_version="huxi")
    opponent = copy.deepcopy(model).eval()
    specs = [(118, 1, 0, -1), (46, 2, 1, 0)]
    rows, results, questions = play(model, [opponent], specs, slots=2, seed=7, training=True)
    again, same_results, same_questions = play(
        model, [opponent], specs, slots=2, seed=7, training=True
    )
    assert not questions and not same_questions
    assert results == same_results and len(results) == 2
    assert len(rows) == len(again) > 0
    for left, right in zip(rows, again, strict=True):
        np.testing.assert_array_equal(left[0], right[0])
        np.testing.assert_array_equal(left[1], right[1])
        assert left[2:] == right[2:]
        assert 0 <= left[2] < len(left[1])
        assert np.isfinite(left[3:]).all()
    before = copy.deepcopy(model.state_dict())
    metrics = ppo_update(
        model,
        opponent,
        torch.optim.AdamW(model.parameters(), lr=1e-4),
        rows,
        random.Random(1),
        device="cpu",
        epochs=1,
        batch_size=32,
    )
    assert metrics["updates"] > 0 and np.isfinite(metrics["loss"])
    assert not torch.equal(
        before["policy_context.weight"], model.state_dict()["policy_context.weight"]
    )


def test_inference_masks_padding_and_preserves_action_availability():
    torch.set_num_threads(1)
    engine = RuleEngine()
    first = engine.new_game(118)
    second = engine.new_game(46)
    while len(engine.legal_actions(second)) == 1:
        second = engine.step(second, engine.legal_actions(second)[0])
    observations = [engine.observation(first, first.turn), engine.observation(second, second.turn)]
    model = PolicyValueNet("mlp")
    selected, *_ = infer(model, observations, "cpu", torch.Generator().manual_seed(1))
    assert all(
        0 <= i < len(obs.legal_actions) for i, obs in zip(selected, observations, strict=True)
    )


def test_evaluation_aggregates_correlated_seats_by_seed_and_rejects_unmatched_games():
    reference = [
        {"seed": s, "seat": seat, "opponent": 0, "cash": 0} for s in (100, 101) for seat in range(3)
    ]
    candidate = [{**r, "cash": 6 if r["seed"] == 100 else -3} for r in reference]
    summary = paired_summary(candidate, reference, 0)["0"]
    assert summary["seeds"] == 2 and summary["games"] == 6
    assert summary["mean_cash_delta"] == 1.5
    with pytest.raises(RuntimeError, match="unmatched"):
        paired_summary(candidate[:-1], reference, 0)
