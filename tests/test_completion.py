"""Completion distance follows intake rules, not arbitrary tile replacements."""

from collections import Counter
from dataclasses import replace

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")
from zimortal.analysis.completion import (
    Position,
    SearchLimit,
    Solver,
    analyze,
    live_upper,
    position,
)
from zimortal.engine import SourceType as S
from zimortal.training.boundary import fixture


@pytest.mark.parametrize(
    "seed,expected", [(240000030, 2), (240000032, 1), (240000008, 1), (280000003, 4)]
)
def test_exact_threshold_distance(seed, expected):
    obs, _target, _state = fixture(seed)
    result = analyze(obs)
    assert result.status == "exact" and result.distance == expected
    assert all(x["distance_after"] == expected - 1 for x in result.effective)
    assert all(x["huxi"] >= 15 for x in result.wins)
    assert all(x["net_payoff"] == 2 * x["amount_each"] for x in result.wins)


def test_copy_exhaustion_and_disabled_state_are_unreachable():
    obs, _target, _ = fixture(240000032)
    assert Solver().solve(position(obs), (0,) * 20).status == "unreachable"
    assert (
        Solver().solve(replace(position(obs), disabled=True), live_upper(obs)).status
        == "unreachable"
    )


def test_limits_never_emit_an_exact_label():
    obs, _target, _ = fixture(240000030)
    with pytest.raises(SearchLimit):
        analyze(obs, max_nodes=1)


def test_solver_uses_own_visible_counts_only_and_respects_kans():
    obs, _target, _ = fixture(240000030)
    solver = Solver(((0, S.DRAW),))
    p = position(obs)
    for tile, n in enumerate(live_upper(obs)):
        if n:
            children, _wins = solver.offer(p, tile, (0, S.DRAW))
            for child in children:
                assert all(Counter(child.hand)[k] == 3 for k in child.kans)
                assert child.kans <= p.kans
    with pytest.raises(ValueError):
        analyze(replace(obs, phase="discard"))
    with pytest.raises(ValueError):
        Solver().solve(Position((), (), frozenset()), (5,) * 20)


def test_empty_wall_cannot_produce_future_winning_draw():
    obs, _target, _ = fixture(240000032)
    assert analyze(replace(obs, remaining_tiles=0)).status == "unreachable"
    result = analyze(replace(obs, remaining_tiles=1))
    assert all(e["live_upper"] <= 1 for e in result.effective)


def test_current_legal_hu_is_distance_zero():
    from zimortal.analysis.completion import analyze_decision
    from zimortal.engine import ActionType as A
    from zimortal.engine import RuleEngine

    _obs, _target, state = fixture(240000032)
    engine = RuleEngine()
    state = engine.step(state, engine.legal_actions(state)[0])
    while engine.legal_actions(state)[0].forced:
        state = engine.step(state, engine.legal_actions(state)[0])
    actions = engine.legal_actions(state)
    assert actions[0].kind == A.HU
    result = analyze_decision(engine.observation(state, 0))
    assert result.distance == 0
    assert result.wins[0]["huxi"] >= 15
