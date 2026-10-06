import random
from collections import Counter

import pytest

from zimortal.engine import (
    Action,
    GameState,
    Meld,
    PendingTile,
    PlayerState,
    RuleEngine,
    base_amount,
    detect_fan,
    enumerate_chi_with_required_bi,
    evaluate_hand,
    is_big,
    is_red,
    make_deck,
    meld_huxi,
    observe,
    rank,
    settle,
    tile_name,
)
from zimortal.engine import (
    ActionType as A,
)
from zimortal.engine import (
    MeldType as M,
)
from zimortal.engine import (
    SourceType as S,
)


@pytest.mark.parametrize("tile", range(20))
def test_tiles(tile):
    assert rank(tile) == tile % 10 + 1
    assert is_big(tile) == (tile >= 10)
    assert is_red(tile) == (tile % 10 in (1, 6, 9))
    assert len(tile_name(tile)) == 1


@pytest.mark.parametrize("tile", [-1, 20, True, 1.2])
def test_invalid_tile(tile):
    with pytest.raises(ValueError):
        tile_name(tile)


def test_deck_deal():
    assert Counter(make_deck(1)) == Counter({t: 4 for t in range(20)})
    assert make_deck(1) == make_deck(1)
    e = RuleEngine()
    for dealer in range(3):
        s = e.new_game(12, dealer)
        assert len(s.deck) == 19
        assert [len(p.hand) for p in s.players] == [21 if i == dealer else 20 for i in range(3)]
        assert s.serialize() == e.new_game(12, dealer).serialize()


@pytest.mark.parametrize(
    "hand,tile,pattern",
    [
        ([2, 3], 4, (2, 3, 4)),
        ([0, 1], 2, (0, 1, 2)),
        ([1, 6], 9, (1, 6, 9)),
        ([0, 10], 0, (0, 0, 10)),
        ([10, 10], 0, (0, 10, 10)),
    ],
)
def test_chi_patterns(hand, tile, pattern):
    assert (pattern, ()) in enumerate_chi_with_required_bi(hand, tile)


def test_complete_bi():
    options = enumerate_chi_with_required_bi([0, 1, 2, 3, 4], 2)
    assert ((0, 1, 2), ((2, 3, 4),)) in options
    assert not enumerate_chi_with_required_bi([0, 1, 2, 2], 2)
    assert not enumerate_chi_with_required_bi([0, 0, 0, 1], 2, protected={0})


@pytest.mark.parametrize(
    "kind,small,big",
    [
        (M.PENG, 1, 3),
        (M.WEI, 3, 6),
        (M.STINKY_WEI, 3, 6),
        (M.KAN, 3, 6),
        (M.PAO, 6, 9),
        (M.TI, 9, 12),
    ],
)
def test_huxi(kind, small, big):
    n = 4 if kind in (M.PAO, M.TI) else 3
    assert meld_huxi(Meld(kind, (0,) * n)) == small
    assert meld_huxi(Meld(kind, (10,) * n)) == big


@pytest.mark.parametrize(
    "tiles,huxi",
    [
        ((0, 1, 2), 3),
        ((10, 11, 12), 6),
        ((1, 6, 9), 3),
        ((11, 16, 19), 6),
        ((2, 3, 4), 0),
        ((0, 0, 10), 0),
    ],
)
def test_chi_huxi(tiles, huxi):
    assert meld_huxi(Meld(M.CHI, tiles)) == huxi


@pytest.mark.parametrize(
    "huxi,amount", [(15, 10), (17, 10), (18, 15), (20, 15), (21, 20), (23, 20), (24, 25)]
)
def test_base_amount(huxi, amount):
    assert base_amount(huxi) == amount


def test_evaluator():
    hand = [t for t in (0, 2, 3, 4, 5, 7, 8) for _ in range(3)]
    wins = evaluate_hand(hand, quad_requires_pair=True)
    assert wins and all(len(w) == 7 for w in wins)
    assert not evaluate_hand(
        [2, 3, 4],
        [
            Meld(M.CHI, tiles)
            for tiles in ((2, 3, 4), (5, 6, 7), (5, 6, 7), (12, 13, 14), (12, 13, 14), (15, 16, 17))
        ],
        quad_requires_pair=True,
    )
    quad_hand = [t for t in (2, 3, 4, 5, 7) for _ in range(3)] + [8, 8]
    assert evaluate_hand(quad_hand, [Meld(M.PAO, (0,) * 4)], quad_requires_pair=True)
    assert not evaluate_hand(quad_hand[:-1], [Meld(M.PAO, (0,) * 4)], quad_requires_pair=True)


def response(hand=(), tile=0, source=S.DRAW, source_player=0, seat=0, melds=()):
    s = GameState(
        [PlayerState() for _ in range(3)],
        [],
        phase="respond",
        pending=PendingTile(tile, source_player, source),
    )
    s.players[seat] = PlayerState(hand=list(hand), melds=list(melds))
    return s


def test_wei_stinky_and_pass_peng():
    e = RuleEngine()
    s = response([0, 0], source=S.DISCARD, source_player=2)
    actions = e.legal_actions(s)
    assert actions[0].kind == A.PENG
    s = e.step(s, actions[1])
    assert 0 in s.players[0].passed_peng
    assert all(a.kind != A.PENG for a in e.legal_actions(s))
    s.pending = PendingTile(0, 0, S.DRAW)
    s.passed.clear()
    actions = e.legal_actions(s)
    assert len(actions) == 1 and actions[0].kind == A.STINKY_WEI and actions[0].forced
    s = e.step(s, actions[0])
    assert meld_huxi(s.players[0].melds[0]) == 3


@pytest.mark.parametrize(
    "hand,melds,source,source_player,expected",
    [
        ([0, 0], [], S.DRAW, 0, A.WEI),
        ([0, 0, 0], [], S.DRAW, 0, A.TI),
        ([0, 0, 0], [], S.DRAW, 1, A.PAO),
        ([0, 0, 0], [], S.DISCARD, 1, A.PAO),
        ([], [Meld(M.WEI, (0,) * 3)], S.DRAW, 0, A.TI),
        ([], [Meld(M.WEI, (0,) * 3)], S.DISCARD, 1, A.PAO),
        ([], [Meld(M.PENG, (0,) * 3)], S.DRAW, 0, A.PAO),
    ],
)
def test_forced_actions(hand, melds, source, source_player, expected):
    e = RuleEngine()
    s = response(hand, source=source, source_player=source_player, melds=melds)
    (action,) = e.legal_actions(s)
    assert action.kind == expected and action.forced
    out = e.step(s, action)
    assert len(out.players[0].melds) == 1
    assert out.players[0].melds[0].kind.value == expected.value
    assert len(out.players[0].melds[0].tiles) == (4 if expected in (A.PAO, A.TI) else 3)


def test_peng_discard_cannot_upgrade():
    s = response(melds=[Meld(M.PENG, (0,) * 3)], source=S.DISCARD, source_player=1)
    assert all(a.kind != A.PAO for a in RuleEngine().legal_actions(s))


def test_jiabi():
    p = PlayerState()
    assert p.needs_discard_after(A.PAO)
    assert not p.needs_discard_after(A.TI)
    assert not p.needs_discard_after(A.PAO)
    for kind in (A.CHI, A.PENG, A.WEI, A.PAO, A.TI):
        p = PlayerState(quad_count=2, opening_double_ti_pending=True)
        assert not p.needs_discard_after(kind)
        assert not p.opening_double_ti_pending


def test_hu_only_draw_and_priority():
    e = RuleEngine()
    hand = [t for t in (0, 2, 3, 4, 5, 7, 8) for _ in range(3)]
    hand.remove(0)
    s = response(hand, source_player=1, seat=0)
    assert e.legal_actions(s)[0].kind == A.HU
    s.pending = PendingTile(0, 1, S.DISCARD)
    assert all(a.kind != A.HU for a in e.legal_actions(s))
    s.pending = PendingTile(0, 0, S.DRAW)
    assert e.legal_actions(s)[0].kind == A.WEI
    out = e.step(s, e.legal_actions(s)[0])
    assert e.legal_actions(out)[0].kind == A.HU
    out = e.step(out, e.legal_actions(out)[1])
    assert out.phase == "discard"


def test_hu_before_pao_and_run_hu():
    e = RuleEngine()
    s = response(
        [t for t in (2, 3, 4, 5, 7) for _ in range(3)] + [8, 8],
        melds=[Meld(M.PENG, (0,) * 3)],
        source_player=1,
    )
    assert e.legal_actions(s)[0].kind == A.HU
    out = e.step(s, e.legal_actions(s)[1])
    assert e.legal_actions(out)[0].kind == A.PAO
    out = e.step(out, e.legal_actions(out)[0])
    assert out.players[0].melds[0].kind == M.PAO
    assert meld_huxi(out.players[0].melds[0]) == 6


def test_chi_pass_and_peng_priority():
    e = RuleEngine()
    s = response([0, 1, 8], tile=2, source=S.DISCARD, source_player=2)
    assert e.legal_actions(s)[0].kind == A.CHI
    out = e.step(s, e.legal_actions(s)[-1])
    assert 2 in out.players[0].passed_chi
    s.players[1].hand = [2, 2]
    assert e.legal_actions(s)[0].kind == A.PENG


def test_heavenly_and_settlement_roundtrip():
    e = RuleEngine()
    hand = [t for t in (0, 2, 3, 4, 5, 7, 8) for _ in range(3)]
    s = GameState([PlayerState(hand), PlayerState(), PlayerState()], [], phase="opening")
    out = e.step(s, e.legal_actions(s)[0])
    assert out.terminal and out.settlement.fan["天胡"] == 5
    assert sum(out.settlement.payments) == 0
    assert GameState.deserialize(out.serialize()) == out


@pytest.mark.parametrize("seed", range(20))
def test_full_game_conservation_replay_observation(seed):
    e = RuleEngine()
    initial = e.new_game(seed)
    s = initial
    rng = random.Random(seed)
    for _ in range(1000):
        if s.terminal:
            break
        legal = e.legal_actions(s)
        assert legal, (seed, s.phase)
        obs = observe(e, s, 0)
        assert not hasattr(obs, "deck")
        assert not hasattr(obs.players[1], "hand")
        s.validate()
        before = s.serialize()
        s = e.step(s, rng.choice(legal))
        counts = Counter(
            s.deck
            + s.river
            + [t for p in s.players for t in p.hand]
            + [t for p in s.players for m in p.melds for t in m.tiles]
        )
        if s.pending:
            counts[s.pending.tile] += 1
        assert counts == Counter({t: 4 for t in range(20)})
        assert GameState.deserialize(before).serialize() == before
    assert s.terminal
    s.validate()
    assert e.replay(initial, s.history).serialize() == s.serialize()


@pytest.mark.parametrize(
    "name,groups,value",
    [
        ("黑胡", [Meld(M.KAN, (0,) * 3)], 5),
        ("点胡", [Meld(M.CHI, (0, 1, 2))], 4),
        ("三扁", [Meld(M.KAN, (1,) * 3)], 3),
        ("四扁", [Meld(M.TI, (1,) * 4)], 4),
        ("双漂", [Meld(M.WEI, (1,) * 3), Meld(M.PENG, (6,) * 3)], 2),
        ("十红", [Meld(M.TI, (1,) * 4), Meld(M.PENG, (6,) * 3), Meld(M.WEI, (9,) * 3)], 2),
        ("十八大", [Meld(M.KAN, (t,) * 3) for t in range(10, 16)], 5),
        ("十八小", [Meld(M.KAN, (t,) * 3) for t in range(6)], 5),
        ("碰碰胡", [Meld(M.KAN, (0,) * 3)], 5),
    ],
)
def test_fan(name, groups, value):
    assert detect_fan(groups)[name] == value


def test_fan_additive_and_ordinary():
    groups = [Meld(M.KAN, (t,) * 3) for t in (10, 12, 13, 14, 15, 17)]
    assert detect_fan(groups) == {"黑胡": 5, "十八大": 5, "碰碰胡": 5}
    result = settle(1, groups, ordinary_fan=1)
    assert result.amount_each == base_amount(36) * 15
    assert result.payments == (-result.amount_each, result.amount_each * 2, -result.amount_each)
    assert detect_fan([Meld(M.CHI, (0, 1, 2))], earthly=True)["地胡"] == 5


def test_multi_hu_order(monkeypatch):
    e = RuleEngine()
    s = response(source_player=1)
    # Isolate the arbitration contract; decomposition has separate exact-cover tests.
    monkeypatch.setattr(e, "_wins", lambda state, player: (True,))
    assert e.legal_actions(s)[0].player == 1
    s = e.step(s, e.legal_actions(s)[1])
    assert e.legal_actions(s)[0].player == 2
    s = e.step(s, e.legal_actions(s)[1])
    assert e.legal_actions(s)[0].player == 0


def test_pass_peng_does_not_pass_chi():
    e = RuleEngine()
    s = response([0, 1, 1, 2, 2, 3, 3, 4, 8], tile=2, source=S.DISCARD, source_player=2)
    assert e.legal_actions(s)[0].kind == A.PENG
    s = e.step(s, e.legal_actions(s)[1])
    assert e.legal_actions(s)[0].kind == A.CHI


def test_earthly_context():
    e = RuleEngine()
    hand = [t for t in (0, 2, 3, 4, 5, 7, 8) for _ in range(3)]
    hand.remove(0)
    s = response(hand, source_player=2, seat=1)
    s.history.append(Action(A.DRAW, 2, forced=True))
    out = e.step(s, e.legal_actions(s)[0])
    assert out.settlement.fan["地胡"] == 5
    s.history.append(Action(A.CHI, 1))
    out = e.step(s, e.legal_actions(s)[0])
    assert "地胡" not in out.settlement.fan


def test_opening_double_ti():
    e = RuleEngine()
    s = GameState(
        [PlayerState([0] * 4 + [2] * 4 + [3, 4]), PlayerState(), PlayerState()], [], phase="opening"
    )
    for tile in (0, 2):
        (action,) = e.legal_actions(s)
        assert action.kind == A.TI and action.tile == tile
        s = e.step(s, action)
    assert s.players[0].opening_double_ti_pending
    assert s.players[0].quad_count == 2
    assert not s.players[0].kans


def test_observation_does_not_depend_on_hidden_order():
    e = RuleEngine()
    s = e.new_game(8)
    o = observe(e, s, 0)
    other = s.clone()
    other.deck.reverse()
    other.players[1].hand.reverse()
    assert observe(e, other, 0) == o
    other.players[1].passed_peng.add(9)
    assert observe(e, other, 0) == o


def test_illegal_step_no_mutation():
    e = RuleEngine()
    s = e.new_game(8)
    original = s.serialize()
    with pytest.raises(ValueError):
        e.step(s, Action(A.DISCARD, 1, 0))
    assert s.serialize() == original


@pytest.mark.parametrize("kind", [A.TI, A.PAO])
def test_second_quad_skips_discard(kind):
    e = RuleEngine()
    s = response([0] * 3, source_player=0 if kind == A.TI else 1)
    s.players[0].quad_count = 1
    s = e.step(s, e.legal_actions(s)[0])
    if s.phase == "post_meld":
        s = e.step(s, e.legal_actions(s)[0])
    assert s.phase == "draw" and s.turn == 1


def test_ordinary_fan_one():
    groups = [
        Meld(M.CHI, (0, 1, 2)),
        Meld(M.CHI, (10, 11, 12)),
        Meld(M.CHI, (1, 6, 9)),
        Meld(M.CHI, (11, 16, 19)),
        Meld(M.CHI, (3, 4, 5)),
        Meld(M.CHI, (13, 14, 15)),
        Meld(M.CHI, (7, 8, 9)),
    ]
    assert not detect_fan(groups)
    result = settle(0, groups, ordinary_fan=1)
    assert result.amount_each == 15


def test_multiple_quads_literal_structure():
    quads = [Meld(M.PAO, (0,) * 4), Meld(M.TI, (10,) * 4)]
    hand = [t for t in (2, 3, 4, 5) for _ in range(3)] + [8, 8]
    wins = evaluate_hand(hand, quads, quad_requires_pair=True)
    assert wins and all(len(w) == 7 for w in wins)


@pytest.mark.parametrize("tiles", [(0, 1), (0, 0, 1), (0, 5, 9), (-1, -1, -1)])
def test_invalid_meld(tiles):
    with pytest.raises(ValueError):
        Meld(M.CHI, tiles)


def test_evaluator_rejects_fifth_copy():
    with pytest.raises(ValueError):
        evaluate_hand([0, 0], [Meld(M.PENG, (0,) * 3)], quad_requires_pair=True)


def test_ti_precedes_hu():
    e = RuleEngine()
    s = response([0] * 3 + [t for t in (2, 3, 4, 5, 7) for _ in range(3)] + [8, 8])
    (action,) = e.legal_actions(s)
    assert action.kind == A.TI
    s = e.step(s, action)
    assert e.legal_actions(s)[0].kind == A.HU


def test_no_red_threshold_false_positives():
    groups = [Meld(M.WEI, (1,) * 3), Meld(M.CHI, (5, 6, 7))]
    fan = detect_fan(groups)
    assert "三扁" not in fan and "双漂" not in fan and "碰碰胡" not in fan


@pytest.mark.parametrize("kind", [M.PAO, M.TI])
def test_fan_threshold_increments(kind):
    groups = [Meld(kind, (1,) * 4), Meld(M.WEI, (6,) * 3), Meld(kind, (9,) * 4)]
    assert detect_fan(groups)["十红"] == 3
    groups = [Meld(M.KAN, (t,) * 3) for t in (10, 12, 13, 14, 15)] + [Meld(kind, (17,) * 4)]
    assert detect_fan(groups)["十八大"] == 6


def test_decision_player_and_observation():
    e = RuleEngine()
    s = response([0, 0], source=S.DISCARD, source_player=2)
    assert e.decision_player(s) == 0
    assert e.observation(s, 0).decision_player == 0
    s.phase = "terminal"
    assert e.decision_player(s) is None
    assert not e.legal_actions(s)


def test_multiple_bi_exhausts_every_copy():
    hand = [0, 1, 1, 2, 2, 3, 3, 4]
    for chi, bi in enumerate_chi_with_required_bi(hand, 2):
        counts = Counter(hand)
        consumed = Counter(chi)
        consumed[2] -= 1
        consumed.update(t for group in bi for t in group)
        assert all(consumed[t] <= counts[t] for t in consumed)
        assert counts[2] == consumed[2]


def test_stinky_wei_before_hu():
    e = RuleEngine()
    hand = [t for t in (0, 2, 3, 4, 5, 7, 8) for _ in range(3)]
    hand.remove(0)
    s = response(hand)
    s.players[0].passed_peng.add(0)
    (action,) = e.legal_actions(s)
    assert action.kind == A.STINKY_WEI
    s = e.step(s, action)
    assert e.legal_actions(s)[0].kind == A.HU


def test_passed_chi_still_can_hu():
    e = RuleEngine()
    hand = [t for t in (0, 2, 3, 4, 5, 7, 8) for _ in range(3)]
    hand.remove(0)
    s = response(hand, source_player=2)
    s.players[0].passed_chi.add(0)
    assert e.legal_actions(s)[0].kind == A.HU


def test_conservation_validator_rejects_missing_tile():
    s = RuleEngine().new_game(1)
    s.validate()
    s.deck.pop()
    with pytest.raises(ValueError):
        s.validate()


def test_unspecified_discard_boundary_is_explicit():
    from zimortal.engine import RuleClarificationRequired

    s = GameState(
        [PlayerState([18] * 3, kans={18}), PlayerState(), PlayerState()], [], phase="discard"
    )
    with pytest.raises(RuleClarificationRequired, match="protected kans"):
        RuleEngine().legal_actions(s)


def test_chi_cannot_leave_only_kans_or_empty():
    e = RuleEngine()
    for hand, kans in [([0, 1], set()), ([0, 1, 8, 8, 8], {8}), ([0, 1, 1, 2, 2, 3, 3, 4], set())]:
        s = response(hand, tile=2, source=S.DISCARD, source_player=2)
        s.players[0].kans = kans
        assert all(a.kind != A.CHI for a in e.legal_actions(s))


def test_chi_empty_hand_allowed_by_opening_double_ti():
    s = response([0, 1], tile=2, source=S.DISCARD, source_player=2)
    s.players[0].opening_double_ti_pending = True
    e = RuleEngine()
    s = e.step(s, e.legal_actions(s)[0])
    assert s.phase == "draw" and s.turn == 1


@pytest.mark.parametrize("stinky", [False, True])
@pytest.mark.parametrize("kan", [False, True])
def test_wei_no_discard_disables_hu(stinky, kan):
    e = RuleEngine()
    s = response([0, 0] + ([8] * 3 if kan else []))
    if kan:
        s.players[0].kans = {8}
    if stinky:
        s.players[0].passed_peng.add(0)
    s = e.step(s, e.legal_actions(s)[0])
    assert s.players[0].hu_disabled
    assert e.legal_actions(s)[0].kind == A.PASS
    s = e.step(s, e.legal_actions(s)[0])
    assert s.phase == "draw" and s.turn == 1
    assert GameState.deserialize(s.serialize()) == s
    s.pending = PendingTile(0, 0, S.DRAW)
    s.phase = "respond"
    assert e.legal_actions(s)[0].kind == A.TI


def test_disabled_player_cannot_chi_peng_hu():
    e = RuleEngine()
    s = response([0, 0, 1, 10, 12], tile=0, source=S.DISCARD, source_player=2)
    s.players[0].hu_disabled = True
    assert all(a.kind not in (A.CHI, A.PENG, A.HU) for a in e.legal_actions(s))


def test_no_discard_wei_still_offers_existing_hu():
    e = RuleEngine()
    hand = [0, 0] + [t for t in (2, 3, 4, 5, 7, 8) for _ in range(3)]
    s = response(hand)
    s.players[0].kans = {2, 3, 4, 5, 7, 8}
    s = e.step(s, e.legal_actions(s)[0])
    assert not s.players[0].hu_disabled
    assert e.legal_actions(s)[0].kind == A.HU
    s = e.step(s, e.legal_actions(s)[1])
    assert s.players[0].hu_disabled
    assert s.phase == "draw" and s.turn == 1


def test_disabled_forced_ti_waives_discard():
    e = RuleEngine()
    s = response([0, 0, 0])
    s.players[0].hu_disabled = True
    s.players[0].kans = {0}
    s = e.step(s, e.legal_actions(s)[0])
    assert s.players[0].melds[0].kind == M.TI
    s = e.step(s, e.legal_actions(s)[0])
    assert s.phase == "draw" and s.turn == 1
    assert e.observation(s, 0).hu_disabled
