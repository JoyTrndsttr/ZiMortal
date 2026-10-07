"""Hu-xi modelling regressions from human review, not a hard-coded policy."""

from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
from zimortal.engine import Action, Meld, RuleEngine, SourceType
from zimortal.engine import ActionType as A
from zimortal.engine import MeldType as M
from zimortal.engine.observation import PublicPlayer
from zimortal.engine.scoring import settle
from zimortal.engine.types import PendingTile
from zimortal.model.encoding import HUXI_CHANNELS, encode_observation
from zimortal.training.valuation import action_scores, draw_values, formed_huxi

HAND17 = (2, 3, 4, 11, 12, 13, 13, 14, 15, 16, 16, 17, 17, 17)
MELDS17 = (Meld(M.PENG, (9,) * 3), Meld(M.CHI, (6, 7, 8)))


def claim_observation():
    obs = RuleEngine().observation(RuleEngine().new_game(1), 1)
    players = list(obs.players)
    players[1] = PublicPlayer(len(HAND17), MELDS17)
    actions = (
        Action(A.PENG, 1, 13, source_player=2, source_type=SourceType.DISCARD),
        Action(A.PASS, 1, 13, source_player=2, source_type=SourceType.DISCARD),
    )
    return replace(
        obs,
        hand=HAND17,
        players=tuple(players),
        pending=PendingTile(13, 2, SourceType.DISCARD),
        phase="respond",
        legal_actions=actions,
    )


def test_structural_waits_below_fifteen_cannot_win():
    values = draw_values(HAND17, MELDS17, (17,))
    assert {t for t, (hu, _amount, _fan) in enumerate(values) if hu >= 0} == {6, 9, 16, 17}
    assert max(hu for hu, _amount, _fan in values) == 13
    assert all(amount == 0 for _hu, amount, _fan in values)


def test_teacher_prefers_peng_to_raise_huxi():
    scores = action_scores(claim_observation())
    assert scores[0] > scores[1]


def test_scoring_groups_cannot_double_count_shared_red_tile():
    assert formed_huxi((10, 11, 12, 16, 19), (), ()) == 6
    assert formed_huxi((10, 11, 11, 12, 16, 19), (), ()) == 12
    assert formed_huxi((10, 11, 11, 11, 12, 16, 19), (), (11,)) == 6


def test_discarding_big_one_loses_six_huxi():
    hand = (2, 3, 4, 6, 7, 10, 11, 12, 13, 13, 14, 15, 16, 16, 17, 17, 17, 18)
    melds = (Meld(M.PENG, (9,) * 3),)
    assert formed_huxi(hand, melds, (17,)) == 13
    after = list(hand)
    after.remove(10)
    assert formed_huxi(after, melds, (17,)) == 7
    obs = claim_observation()
    players = list(obs.players)
    players[1] = PublicPlayer(len(hand), melds)
    legal = tuple(Action(A.DISCARD, 1, t) for t in sorted(set(hand) - {17}))
    obs = replace(
        obs, hand=hand, players=tuple(players), pending=None, phase="discard", legal_actions=legal
    )
    scores = action_scores(obs)
    assert legal[int(np.argmax(scores))].tile not in (10, 11, 12)
    encoded = encode_observation(obs, "huxi")
    assert encoded.shape == (HUXI_CHANNELS, 20)
    assert encoded[48, 10] == 0.5


def test_settlement_reward_includes_all_fan_and_is_zero_sum():
    from zimortal.training.reinforce import terminal_reward

    kans = (2, 3, 4, 5, 7, 8)
    hand = (0, 0) + tuple(t for t in kans for _ in range(3))
    outcome = draw_values(hand, (), kans)[0]
    groups = (Meld(M.WEI, (0,) * 3),) + tuple(Meld(M.KAN, (t,) * 3) for t in kans)
    settlement = settle(0, groups, ordinary_fan=1)
    assert outcome == (settlement.huxi, settlement.amount_each, sum(settlement.fan.values()))
    rewards = [terminal_reward(settlement, i, "settlement") for i in range(3)]
    assert sum(rewards) == pytest.approx(0)
    assert rewards[0] == 2 * settlement.amount_each / 100
    assert rewards[0] > 1  # Monetary rewards must not be clipped to win/loss.


def test_huxi_model_roundtrip_and_hidden_information(tmp_path):
    from zimortal.model.network import PolicyValueNet
    from zimortal.training.runtime import load_model, save_model

    model = PolicyValueNet(feature_version="huxi")
    path = tmp_path / "model.pt"
    save_model(model, path)
    restored = load_model(path)
    assert restored.feature_version == "huxi" and restored.input_channels == 55
    engine = RuleEngine()
    state = engine.new_game(118)
    changed = state.clone()
    changed.deck.reverse()
    changed.players[1].hand[0], changed.deck[0] = changed.deck[0], changed.players[1].hand[0]
    np.testing.assert_array_equal(
        encode_observation(engine.observation(state, 0), "huxi"),
        encode_observation(engine.observation(changed, 0), "huxi"),
    )
