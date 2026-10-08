"""Exact optimistic legal-intake distance, not replacement/perturbation distance.

Opponents do not compete. Future offers can be own draws, upstream draws or
upstream discards. A final winning draw counts as one intake. Unclaimed offers
are omitted: they cannot improve our hand or relax a pass restriction. The
physical unknown-copy upper bound is consumed by every accepted offer.
"""

from collections import Counter
from dataclasses import asdict, dataclass, replace
from functools import lru_cache

from zimortal.engine import ActionType as A
from zimortal.engine import GameState, PlayerState, RuleEngine
from zimortal.engine import SourceType as S
from zimortal.engine.scoring import settle
from zimortal.engine.types import PendingTile

SOURCES = ((0, S.DRAW), (2, S.DRAW), (2, S.DISCARD))


class SearchLimit(RuntimeError):
    """No exact label may be emitted after a resource limit."""


@dataclass(frozen=True)
class Position:
    hand: tuple
    melds: tuple
    kans: frozenset
    passed_chi: frozenset = frozenset()
    passed_peng: frozenset = frozenset()
    quad_count: int = 0
    waived: bool = False
    disabled: bool = False

    def state(self):
        own = PlayerState(
            list(self.hand),
            list(self.melds),
            set(self.passed_peng),
            set(self.passed_chi),
            set(self.kans),
            self.disabled,
            self.quad_count,
            self.waived,
        )
        return GameState([own, PlayerState(), PlayerState()], [], phase="draw", dealer=0)

    @classmethod
    def of(cls, state):
        p = state.players[0]
        return cls(
            tuple(sorted(p.hand)),
            tuple(sorted(p.melds, key=lambda m: (m.kind.value, m.tiles))),
            frozenset(p.kans),
            frozenset(p.passed_chi),
            frozenset(p.passed_peng),
            p.quad_count,
            p.opening_double_ti_pending,
            p.hu_disabled,
        )


@dataclass(frozen=True)
class Completion:
    distance: int | None
    status: str
    effective: tuple
    wins: tuple
    nodes: int
    definition: str = "minimum accepted legal intakes including final hu; optimistic opponents"

    def json(self):
        return asdict(self)


def live_upper(obs):
    known = Counter(obs.hand) + Counter(obs.river)
    for p in obs.players:
        known.update(t for m in p.melds for t in m.tiles)
    if obs.pending:
        known[obs.pending.tile] += 1
    if any(n > 4 for n in known.values()):
        raise ValueError("observation violates four-copy bound")
    return tuple(max(0, 4 - known[t]) for t in range(20))


class Solver:
    def __init__(self, sources=SOURCES, max_nodes=None):
        if not sources or any(x not in SOURCES for x in sources):
            raise ValueError("unsupported projected offer source")
        self.sources = tuple(sources)
        self.max_nodes = max_nodes
        self.nodes = 0
        self.engine = RuleEngine()
        self.offer = lru_cache(maxsize=None)(self._offer)
        self.terminal = lru_cache(maxsize=None)(self._terminal)
        self.within = lru_cache(maxsize=None)(self._within)

    def _terminal(self, position, tile, source):
        if source[1] != S.DRAW or position.disabled:
            return ()
        state = position.state()
        state.pending = PendingTile(tile, source[0], source[1])
        state.phase = "respond"
        mandatory = position.hand.count(tile) in (2, 3) or any(
            m.tiles[0] == tile and m.kind.value in ("wei", "stinky_wei") for m in position.melds
        )
        if source[0] == 0 and mandatory:
            state = self.engine.step(state, self.engine.legal_actions(state)[0])
        groups = self.engine._wins(state, 0)
        if not groups:
            return ()
        s = max((settle(0, g, ordinary_fan=1) for g in groups), key=lambda x: x.amount_each)
        return (
            {
                "tile": tile,
                "source": "own_draw" if source[0] == 0 else "upstream_draw",
                "huxi": s.huxi,
                "fan": s.fan,
                "amount_each": s.amount_each,
                "net_payoff": s.payments[0],
                "groups": [asdict(g) for g in s.groups],
            },
        )

    def _offer(self, position, tile, source):
        if wins := self.terminal(position, tile, source):
            return (), wins
        state = position.state()
        state.pending = PendingTile(tile, source[0], source[1])
        state.phase = "respond"
        outcomes = []
        wins = []
        # Passing a higher priority own optional stage may expose the chi stage.
        while True:
            actions = self.engine.legal_actions(state)
            if actions[0].kind == A.HU:
                end = self.engine.step(state, actions[0])
                s = end.settlement
                wins.append(
                    {
                        "tile": tile,
                        "source": "own_draw" if source[0] == 0 else "upstream_draw",
                        "huxi": s.huxi,
                        "fan": s.fan,
                        "amount_each": s.amount_each,
                        "net_payoff": s.payments[0],
                        "groups": [asdict(g) for g in s.groups],
                    }
                )
                break
            if state.phase in ("draw", "discard", "terminal"):
                if state.players[0].hu_disabled:
                    break
                if state.phase == "discard":
                    outcomes.extend(Position.of(self.engine.step(state, a)) for a in actions)
                elif state.phase == "draw":
                    outcomes.append(Position.of(state))
                break
            claims = [a for a in actions if a.player == 0 and a.kind not in (A.PASS, A.DRAW)]
            for action in claims:
                after = self.engine.step(state, action)
                if action.forced:
                    state = after
                    break
                discards = self.engine.legal_actions(after)
                if after.phase == "discard":
                    outcomes.extend(Position.of(self.engine.step(after, a)) for a in discards)
                elif after.phase == "draw":
                    outcomes.append(Position.of(after))
                else:
                    raise RuntimeError("unexpected optional intake continuation")
            else:
                passes = [a for a in actions if a.kind == A.PASS]
                if not passes or passes[0].forced and state.phase != "post_meld":
                    break
                state = self.engine.step(state, passes[0])
                continue
            continue
        return tuple(
            sorted(
                set(outcomes),
                key=lambda p: (
                    p.hand,
                    tuple((m.kind.value, m.tiles) for m in p.melds),
                    tuple(sorted(p.passed_chi)),
                    tuple(sorted(p.passed_peng)),
                ),
            )
        ), tuple(wins)

    def _within(self, position, supply, depth):
        if depth <= 0 or position.disabled:
            return False
        self.nodes += 1
        if self.max_nodes is not None and self.nodes > self.max_nodes:
            raise SearchLimit(f"completion search exceeded {self.max_nodes} nodes")
        transitions = []
        for tile, remaining in enumerate(supply):
            if remaining and any(self.terminal(position, tile, source) for source in self.sources):
                return True
        if depth == 1:
            return False
        for tile, remaining in enumerate(supply):
            if not remaining:
                continue
            for source in self.sources:
                children, wins = self.offer(position, tile, source)
                if wins:
                    return True
                if depth > 1 and children:
                    next_supply = list(supply)
                    next_supply[tile] -= 1
                    transitions.extend((p, tuple(next_supply)) for p in children if not p.disabled)
        return any(self.within(p, s, depth - 1) for p, s in transitions)

    def solve(self, position, supply):
        if len(supply) != 20 or any(n < 0 or n > 4 for n in supply):
            raise ValueError("invalid copy upper bounds")
        # Each accepted intake consumes a physical unseen copy. Exhausting this
        # bound proves unreachable; no finite-depth guess is returned as exact.
        for depth in range(1, sum(supply) + 1):
            if not self.within(position, tuple(supply), depth):
                continue
            effective, wins = [], []
            for tile, remaining in enumerate(supply):
                if not remaining:
                    continue
                following = list(supply)
                following[tile] -= 1
                for source in self.sources:
                    children, terminal = self.offer(position, tile, source)
                    if terminal or any(
                        self.within(p, tuple(following), depth - 1) for p in children
                    ):
                        effective.append(
                            {
                                "tile": tile,
                                "source": (
                                    "upstream_draw" if source[1] == S.DRAW else "upstream_discard"
                                )
                                if source[0]
                                else "own_draw",
                                "live_upper": remaining,
                                "distance_after": depth - 1,
                            }
                        )
                    wins.extend(terminal)
            return Completion(depth, "exact", tuple(effective), tuple(wins), self.nodes)
        return Completion(None, "unreachable", (), (), self.nodes)


def position(obs):
    counts = Counter(obs.hand)
    initial_ti = sum(
        a.player == obs.player and a.kind == A.TI and a.source_type == S.INITIAL
        for a in obs.history
    )
    intake = any(
        a.player == obs.player
        and a.kind in (A.CHI, A.PENG, A.WEI, A.STINKY_WEI, A.PAO, A.TI)
        and a.source_type != S.INITIAL
        for a in obs.history
    )
    melds = obs.players[obs.player].melds
    return Position(
        obs.hand,
        melds,
        frozenset(t for t, n in counts.items() if n == 3),
        obs.passed_chi,
        obs.passed_peng,
        sum(m.kind.value in ("pao", "ti") for m in melds),
        initial_ti >= 2 and not intake,
        obs.hu_disabled,
    )


def analyze(obs, *, sources=SOURCES, max_nodes=None):
    """Analyze a waiting hand; pending decisions require executing an action first."""
    if obs.pending is not None or obs.phase not in ("draw",):
        raise ValueError("completion labels require a waiting state after resolving intake/discard")
    if obs.remaining_tiles == 0:
        return Completion(None, "unreachable", (), (), 0)
    result = Solver(sources, max_nodes).solve(position(obs), live_upper(obs))
    return replace(
        result,
        effective=tuple(
            dict(
                e,
                unknown_copy_upper=e["live_upper"],
                live_upper=min(e["live_upper"], obs.remaining_tiles),
            )
            for e in result.effective
        ),
    )


def analyze_decision(obs, *, sources=SOURCES, max_nodes=None):
    """Current hu is zero; required discard costs no new intake.

    Optional/forced pending claims must first be resolved with the real engine;
    this avoids fabricating priority, already-passed-hu or waiver context.
    """
    if any(a.kind == A.HU for a in obs.legal_actions):
        from zimortal.engine.scoring import settle

        state = position(obs).state()
        state.phase = obs.phase
        if obs.pending:
            state.pending = PendingTile(
                obs.pending.tile, (obs.pending.player - obs.player) % 3, obs.pending.source
            )
        groups = RuleEngine()._wins(state, 0)
        values = [settle(0, g, ordinary_fan=1) for g in groups]
        best = max(values, key=lambda s: s.amount_each)
        win = {
            "tile": obs.pending.tile if obs.pending else None,
            "huxi": best.huxi,
            "fan": best.fan,
            "amount_each": best.amount_each,
            "net_payoff": best.payments[0],
            "amount_scope": "ordinary; no heavenly/earthly bonuses",
        }
        return Completion(0, "exact", (), (win,), 0)
    if obs.phase == "draw" and obs.pending is None:
        return analyze(obs, sources=sources, max_nodes=max_nodes)
    if obs.phase != "discard" or obs.pending is not None:
        raise ValueError("resolve current claim/post-meld decision before distance analysis")
    solver = Solver(sources, max_nodes)
    supply = live_upper(obs)
    state = position(obs).state()
    state.phase = "discard"
    solutions = []
    for action in solver.engine.legal_actions(state):
        child = Position.of(solver.engine.step(state, action))
        answer = solver.solve(child, supply)
        if answer.distance is not None:
            solutions.append((action.tile, answer))
    if not solutions:
        return Completion(None, "unreachable", (), (), solver.nodes)
    distance = min(r.distance for _t, r in solutions)
    effective = tuple(
        dict(e, discard=t) for t, r in solutions if r.distance == distance for e in r.effective
    )
    wins = tuple(dict(w, discard=t) for t, r in solutions if r.distance == distance for w in r.wins)
    return Completion(distance, "exact", effective, wins, solver.nodes)
