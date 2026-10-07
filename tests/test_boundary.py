"""Synthetic fixtures must agree with actual engine decisions and settlement."""

from collections import Counter

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from zimortal.engine import ActionType as A
from zimortal.engine import RuleEngine
from zimortal.model.encoding import encode_observation
from zimortal.model.network import PolicyValueNet
from zimortal.training.boundary import fixture, outcome_loss, row
from zimortal.training.runtime import load_model, save_model
from zimortal.training.train import batch
from zimortal.training.valuation import draw_values


@pytest.mark.parametrize("seed,expected", [(240000030, 14), (240000032, 15), (240000008, 16)])
def test_boundary_fixture_matches_real_draw_and_settlement(seed, expected):
    obs, target, state = fixture(seed)
    state.validate()
    engine = RuleEngine()
    assert obs.phase == "draw" and obs.pending is None
    assert obs.legal_actions[0].kind == A.DRAW
    kans = tuple(t for t, n in Counter(obs.hand).items() if n == 3)
    label = draw_values(obs.hand, obs.players[0].melds, kans)[target]
    assert label[0] == expected
    after = engine.step(state, obs.legal_actions[0])
    while (
        (actions := engine.legal_actions(after))[0].forced
        and actions[0].player == 0
        and actions[0].kind in (A.WEI, A.STINKY_WEI, A.TI)
    ):
        after = engine.step(after, actions[0])
    hu = next((a for a in actions if a.player == 0 and a.kind == A.HU), None)
    assert (hu is not None) == (expected >= 15)
    if hu:
        end = engine.step(after, hu)
        assert end.settlement.amount_each == label[1]
        assert sum(end.settlement.payments) == 0
    else:
        assert label[1:] == (0, 0)


def test_boundary_features_do_not_reveal_completion_or_hidden_hands():
    obs, _target, state = fixture(240000032)
    changed = state.clone()
    changed.deck.reverse()
    changed.players[1].hand[0], changed.deck[0] = changed.deck[0], changed.players[1].hand[0]
    changed.validate()
    np.testing.assert_array_equal(
        encode_observation(obs, "huxi"),
        encode_observation(RuleEngine().observation(changed, 0), "huxi"),
    )


def test_absent_draw_labels_do_not_pollute_huxi_regression():
    rows = [row(fixture(240000032)[0])]
    model = PolicyValueNet(feature_version="huxi", auxiliary_version="boundary")
    output = model.forward_aux(*batch(rows)[:3])
    hu = output[3].detach().clone().requires_grad_()
    altered = output[:3] + (hu,) + output[4:]
    loss = outcome_loss(altered, rows)
    loss.backward()
    absent = torch.tensor(rows[0][5][1:] < 0)
    assert absent.any()
    assert torch.all(hu.grad[0, 1:][absent] == 0)
    assert hu.grad[0, 1:][~absent].abs().sum() > 0


def test_new_outcome_heads_roundtrip_without_breaking_legacy_models(tmp_path):
    for version in ("legacy", "boundary"):
        model = PolicyValueNet(feature_version="huxi", auxiliary_version=version)
        path = tmp_path / f"{version}.pt"
        save_model(model, path)
        loaded = load_model(path)
        assert loaded.auxiliary_version == version
        rows = [row(fixture(240000032)[0])]
        inputs = batch(rows)[:3]
        with torch.no_grad():
            before = model.forward_aux(*inputs)
            after = loaded.forward_aux(*inputs)
        for a, b in zip(before, after, strict=True):
            if a is None:
                assert b is None
            else:
                torch.testing.assert_close(a, b)


def test_paired_payoff_interval_clusters_seats_and_preserves_difference():
    from zimortal.training.boundary_review import paired_interval

    parent = [
        {"seed": seed, "seat": seat, "payoff": seat * 5} for seed in (1, 2, 3) for seat in range(3)
    ]
    challenger = [r | {"payoff": r["payoff"] + 10} for r in parent]
    result = paired_interval(parent, challenger)
    assert result["seed_clusters"] == 3
    assert result["mean_payoff_difference"] == 10
    assert result["paired_bootstrap_95_percent"] == [10, 10]
    with pytest.raises(ValueError):
        paired_interval(parent, challenger[:3])
