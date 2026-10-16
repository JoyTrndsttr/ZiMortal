import dataclasses
import hashlib

import numpy as np
import pytest
import torch

from zimortal.engine import Action, ActionType
from zimortal.engine.observation import Observation, PublicPlayer
from zimortal.model.encoding import ACTION_DIM, encode_action, encode_observation
from zimortal.model.mortal_faithful import MortalFaithfulQNet
from zimortal.training.mortal_credit_diagnostics import (
    coarse_keys,
    coverage,
    encoded_key,
    exact_observation,
    gradient_probe,
    group_variance,
    scales,
)
from zimortal.training.mortal_offline import tensors
from zimortal.training.mortal_sarsa import parameter_hash


def observation():
    actions = (Action(ActionType.DISCARD, 0, 0), Action(ActionType.DISCARD, 0, 1))
    return Observation(
        0,
        (0, 1, 2),
        False,
        (PublicPlayer(3, ()),) * 3,
        19,
        (),
        (),
        None,
        0,
        0,
        "discard",
        frozenset(),
        frozenset(),
        actions,
    )


def test_exact_hash_keeps_unencoded_public_information_and_legal_set_order_invariance():
    obs = observation()
    prefix = hashlib.sha256().digest()
    changed = dataclasses.replace(obs, phase="other-phase")
    key, legal = exact_observation(obs, prefix)
    assert key != exact_observation(changed, prefix)[0]
    assert key != exact_observation(obs, hashlib.sha256(b"public history").digest())[0]
    reordered = dataclasses.replace(obs, legal_actions=obs.legal_actions[::-1])
    assert (key, legal) == exact_observation(reordered, prefix)
    x = encode_observation(obs, "huxi")
    a = np.stack([encode_action(v, 0) for v in obs.legal_actions])
    assert np.array_equal(x, encode_observation(changed, "huxi"))
    assert encoded_key(x, a) == encoded_key(x, a[::-1])
    assert coarse_keys(x, "discard", 2, legal) == coarse_keys(x.copy(), "discard", 2, legal)


def test_group_variance_does_not_fake_zero_noise_for_singletons_or_leak_same_game():
    key = np.array([0, 0, 0, 1])
    cash = np.array([0.0, 0.0, 10.0, 20.0])
    game = np.array([1, 1, 2, 3])
    r = group_variance(key, cash, game)
    assert r["cross_game_within_df"] == 1
    assert r["cross_game_within_std_yuan"] == pytest.approx(50**0.5)
    assert r["leave_one_game_out_coverage"] == pytest.approx(0.75)
    # Three eligible row residuals: -10,-10,+10. Excluding just one row would leak game1's zero.
    assert r["leave_one_game_out_residual_yuan"]["std"] == pytest.approx(
        np.std([-10.0, -10.0, 10.0])
    )
    one = group_variance(np.arange(4), cash, game)
    assert one["cross_game_within_std_yuan"] is None
    assert one["leave_one_game_out_coverage"] == 0
    state = coverage(key, np.array([0, 0, 1, 0]), cash, game)
    assert state["multi_action_row_fraction"] == pytest.approx(0.75)
    assert state["state_action_variance"]["cross_game_within_df"] == 0


def test_legal_mean_chosen_loss_gradient_matches_exact_formula_and_padding_zero():
    advantage = torch.tensor([[0.5, 1.0, 2.0, 9.0]], requires_grad=True)
    value = torch.tensor([0.3], requires_grad=True)
    mask = torch.tensor([[True, True, True, False]])
    q = value[:, None] + advantage - advantage.masked_fill(~mask, 0).sum(1)[:, None] / 3
    error = q[0, 1] - 0.2
    loss = 0.5 * error.square()
    ga, gv = torch.autograd.grad(loss, (advantage, value))
    assert gv.item() == pytest.approx(error.item())
    assert ga.tolist()[0] == pytest.approx(
        [-error.item() / 3, 2 * error.item() / 3, -error.item() / 3, 0]
    )
    assert (ga[0, 0] - ga[0, 2]).item() == 0


def test_real_structured_scorer_gradient_probe_has_no_model_bn_or_grad_mutation():
    torch.manual_seed(7)
    model = MortalFaithfulQNet(width=16, blocks=1)
    model.eval()
    before = parameter_hash(model)
    rng = np.random.default_rng(9)
    rows = [
        {
            "features": rng.random((55, 20), dtype=np.float32),
            "actions": np.eye(2, ACTION_DIM, dtype=np.float32),
            "chosen": i % 2,
            "terminal_cash": float(i - 2) / 10,
            "steps_to_terminal": 0,
            "family": "discard" if i % 2 else "chi/pass",
        }
        for i in range(4)
    ]
    for train in (False, True):
        result = gradient_probe(model, rows, tensors, device="cpu", train_mode=train, batch=4)
        assert result["state_hash_unchanged"] and parameter_hash(model) == before
        assert result["direct_derivative_max_abs_error"] < 1e-6
        assert result["all"]["groups"]["value"]["mean_gradient_l2"] > 0
        assert result["all"]["groups"]["advantage"]["mean_gradient_l2"] > 0
        assert abs(result["all"]["last_advantage_bias_gradient"]) < 1e-6
        assert all(p.grad is None for p in model.parameters())
        assert not model.training and not model.brain[1].res_unit[0].training


def test_scales_keep_actual_anchor_scores_and_cash_units():
    r = scales(
        [0.0, 0.02],
        [0.9, 0.1],
        0,
        1,
        10.0,
        scores=[0.3 * np.log(0.9 + 1e-8), 0.02 + 0.3 * np.log(0.1 + 1e-8)],
    )
    assert r["raw_margin_yuan"] == pytest.approx(2)
    assert r["residual_yuan"] == pytest.approx(8)
    assert r["anchor_parent_delta_yuan"] == 0
    assert r["centered_advantage_std_yuan"] == pytest.approx(1)


def test_outcome_blind_distinct_state_sampling_never_changes_when_cash_changes():
    import importlib.util
    from pathlib import Path

    path = Path("scripts/diagnose-mortal-credit-assignment.py")
    spec = importlib.util.spec_from_file_location("credit_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    a, b = module.Sample("toy", capacity=3), module.Sample("toy", capacity=3)
    for index in range(30):
        key = hashlib.sha256(str(index % 10).encode()).digest()
        a.offer(key, [index], {"locator": [index], "cash": index})
        b.offer(key, [index], {"locator": [index], "cash": -1000 * index})
    assert [r["locator"] for r in a.result()] == [r["locator"] for r in b.result()]
    assert len(a.result()) == 3


def test_independent_coverage_does_not_count_two_actions_inside_one_game_as_two_outcomes():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "credit_audit", Path("scripts/audit-mortal-credit-assignment.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.independent_coverage(
        np.array([0, 0, 0, 1, 1]), np.array([0, 1, 0, 0, 1]), np.array([1, 1, 1, 2, 3])
    )
    assert result["cross_game_multi_action_states"] == 1
    assert result["cross_game_multi_action_row_fraction"] == pytest.approx(0.4)
    assert result["visit_histogram_states"]["2..2"] == 1
    assert result["visit_histogram_states"]["3..3"] == 1
