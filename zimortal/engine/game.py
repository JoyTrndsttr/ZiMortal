"""Deterministic engine. Policies choose only from the returned legal actions."""

from collections import Counter
from dataclasses import dataclass

from .chi import enumerate_chi_with_required_bi
from .evaluator import evaluate_hand
from .scoring import settle
from .tiles import make_deck
from .types import Action, GameState, Meld, PendingTile, PlayerState
from .types import ActionType as A
from .types import MeldType as M
from .types import SourceType as S


class RuleClarificationRequired(RuntimeError):
    """A reachable boundary whose rule has not yet been specified."""


@dataclass(frozen=True)
class RuleConfig:
    """Rules confirmed by the maintainer; multiple quads retain six groups + pair."""

    hand_sizes: tuple[int, int, int] = (21, 20, 20)
    quad_requires_pair: bool = True
    protect_kans: bool = True
    ordinary_fan: int = 1

    def __post_init__(self):
        if (
            len(self.hand_sizes) != 3
            or any(n < 0 for n in self.hand_sizes)
            or sum(self.hand_sizes) > 80
        ):
            raise ValueError("invalid deal sizes")
        if self.ordinary_fan < 1:
            raise ValueError("ordinary_fan must be positive")


class RuleEngine:
    def __init__(self, config: RuleConfig | None = None):
        self.config = config or RuleConfig()

    def new_game(self, seed=None, dealer=0):
        if dealer not in range(3):
            raise ValueError("invalid dealer")
        deck = make_deck(seed)
        players = []
        for seat in range(3):
            size = self.config.hand_sizes[(seat - dealer) % 3]
            hand = [deck.pop() for _ in range(size)]
            players.append(
                PlayerState(hand=sorted(hand), kans={t for t, n in Counter(hand).items() if n == 3})
            )
        state = GameState(players, deck, turn=dealer, dealer=dealer, phase="opening", seed=seed)
        return state

    def _wins(self, state, player):
        p = state.players[player]
        if p.hu_disabled:
            return ()
        pending = state.pending
        if state.phase in ("opening", "post_meld"):
            return evaluate_hand(
                p.hand,
                p.melds,
                quad_requires_pair=self.config.quad_requires_pair,
                protected=p.kans if self.config.protect_kans else (),
            )
        if pending is None or pending.source != S.DRAW:
            return ()
        hand, melds, kans = list(p.hand), list(p.melds), set(p.kans)
        tile = pending.tile
        # Running the offered fourth tile can itself complete a winning structure.
        upgrade = next(
            (
                i
                for i, m in enumerate(melds)
                if m.tiles[0] == tile and m.kind in (M.PENG, M.WEI, M.STINKY_WEI)
            ),
            None,
        )
        if upgrade is not None:
            melds[upgrade] = Meld(M.PAO, (tile,) * 4)
        elif hand.count(tile) == 3:
            for _ in range(3):
                hand.remove(tile)
            kans.discard(tile)
            melds.append(Meld(M.PAO, (tile,) * 4))
        else:
            hand.append(tile)
        return evaluate_hand(
            hand,
            melds,
            quad_requires_pair=self.config.quad_requires_pair,
            protected=kans if self.config.protect_kans else (),
            exposed_triplet=tile if player != pending.player and p.hand.count(tile) == 2 else None,
        )

    def decision_player(self, state):
        """The player who can act now; separate from the offered tile's source."""
        actions = self.legal_actions(state)
        return actions[0].player if actions else None

    def observation(self, state, player):
        from .observation import observe

        return observe(self, state, player)

    def legal_actions(self, state, player=None):
        actions = self._legal(state)
        return tuple(a for a in actions if player is None or a.player == player)

    def _claim_can_finish_turn(self, player, tile, chi, bi):
        if player.opening_double_ti_pending:
            return True
        remaining = Counter(player.hand)
        used = Counter(chi)
        used[tile] -= 1
        used.update(t for group in bi for t in group)
        remaining.subtract(used)
        protected = player.kans if self.config.protect_kans else set()
        return any(n > 0 and t not in protected for t, n in remaining.items())

    def _legal(self, state):
        if state.terminal:
            return ()
        # Opening quads remain in hand until a visible, replayable forced action.
        for seat in ((state.turn + i) % 3 for i in range(3)):
            for tile, n in sorted(Counter(state.players[seat].hand).items()):
                if n == 4:
                    return (Action(A.TI, seat, tile, source_type=S.INITIAL, forced=True),)
        if state.phase == "opening":
            seat = state.dealer
            if self._wins(state, seat):
                return (
                    Action(A.HU, seat, source_type=S.INITIAL),
                    Action(A.PASS, seat, source_type=S.INITIAL),
                )
            return (Action(A.PASS, seat, source_type=S.INITIAL, forced=True),)
        if state.phase == "post_meld":
            seat = state.turn
            if self._wins(state, seat):
                return (
                    Action(A.HU, seat, source_type=S.DRAW),
                    Action(A.PASS, seat, source_type=S.DRAW),
                )
            return (Action(A.PASS, seat, source_type=S.DRAW, forced=True),)
        if state.phase == "draw":
            return (Action(A.DRAW, state.turn, forced=True),)
        if state.phase == "discard":
            p = state.players[state.turn]
            tiles = set(p.hand) - (p.kans if self.config.protect_kans else set())
            if not tiles:
                raise RuleClarificationRequired(
                    "discard required but only protected kans remain; clarify whether "
                    "the preceding claim is illegal or this discard is waived"
                )
            return tuple(Action(A.DISCARD, state.turn, t) for t in sorted(tiles))
        if state.phase != "respond" or state.pending is None:
            raise ValueError("invalid decision phase")
        q = state.pending
        tile = q.tile
        p = state.players[q.player]
        if q.source == S.DRAW:
            own = next(
                (m for m in p.melds if m.tiles[0] == tile and m.kind in (M.WEI, M.STINKY_WEI)), None
            )
            if p.hand.count(tile) == 3 or own:
                return (
                    Action(
                        A.TI,
                        q.player,
                        tile,
                        source_player=q.player,
                        source_type=q.source,
                        forced=True,
                    ),
                )
            if p.hand.count(tile) == 2:
                kind = A.STINKY_WEI if tile in p.passed_peng else A.WEI
                return (
                    Action(
                        kind,
                        q.player,
                        tile,
                        source_player=q.player,
                        source_type=q.source,
                        forced=True,
                    ),
                )
        order = [(q.player + i) % 3 for i in range(3)]
        if q.source == S.DRAW:
            for seat in order:
                if seat not in state.hu_passed and self._wins(state, seat):
                    return (
                        Action(A.HU, seat, tile, source_player=q.player, source_type=q.source),
                        Action(A.PASS, seat, tile, source_player=q.player, source_type=q.source),
                    )
        for seat in order:
            p = state.players[seat]
            eligible = any(
                m.tiles[0] == tile
                and (m.kind in (M.WEI, M.STINKY_WEI) or (m.kind == M.PENG and q.source == S.DRAW))
                for m in p.melds
            )
            if eligible or p.hand.count(tile) == 3:
                return (
                    Action(
                        A.PAO, seat, tile, source_player=q.player, source_type=q.source, forced=True
                    ),
                )
        for seat in order:
            p = state.players[seat]
            if (
                seat != q.player
                and not p.hu_disabled
                and tile not in p.passed_peng
                and p.hand.count(tile) == 2
                and self._claim_can_finish_turn(p, tile, (tile,) * 3, ())
            ):
                return (
                    Action(A.PENG, seat, tile, source_player=q.player, source_type=q.source),
                    Action(A.PASS, seat, tile, source_player=q.player, source_type=q.source),
                )
        # A draw is offered to its drawer first, then to the drawer's next
        # player. A discard may only be eaten by the discarder's next player.
        chi_order = (q.player, (q.player + 1) % 3) if q.source == S.DRAW else ((q.player + 1) % 3,)
        for seat in chi_order:
            p = state.players[seat]
            if not p.hu_disabled and tile not in p.passed_chi:
                options = enumerate_chi_with_required_bi(
                    p.hand, tile, protected=p.kans if self.config.protect_kans else ()
                )
                options = tuple(
                    (chi, bi)
                    for chi, bi in options
                    if self._claim_can_finish_turn(p, tile, chi, bi)
                )
                if options:
                    return tuple(
                        Action(A.CHI, seat, tile, chi, bi, q.player, q.source)
                        for chi, bi in options
                    ) + (Action(A.PASS, seat, tile, source_player=q.player, source_type=q.source),)
        # Explicit forced pass disposes of the offered tile, advancing the turn.
        return (
            Action(
                A.PASS, q.player, tile, source_player=q.player, source_type=q.source, forced=True
            ),
        )

    def step(self, state, action):
        legal = self.legal_actions(state)
        if action not in legal:
            raise ValueError("illegal action")
        out = state.clone()
        p = out.players[action.player]
        tile = action.tile
        out.history.append(action)
        if action.kind == A.DRAW:
            if not out.deck:
                out.phase = "terminal"
            else:
                out.pending = PendingTile(out.deck.pop(), action.player, S.DRAW)
                out.phase = "respond"
                out.passed.clear()
                out.hu_passed.clear()
        elif action.kind == A.DISCARD:
            p.hand.remove(tile)
            p.passed_chi.add(tile)
            out.pending = PendingTile(tile, action.player, S.DISCARD)
            out.phase = "respond"
            out.passed.clear()
        elif action.kind == A.PASS:
            if out.phase == "opening":
                out.phase = "discard"
                out.turn = out.dealer
            elif out.phase == "post_meld":
                if not (set(p.hand) - (p.kans if self.config.protect_kans else set())):
                    p.hu_disabled = True
                out.phase = out.resume_phase
                out.turn = out.resume_turn
                out.hu_passed.clear()
            elif action.forced:
                out.river.append(tile)
                out.turn = (out.pending.player + 1) % 3
                out.pending = None
                out.passed.clear()
                out.hu_passed.clear()
                out.phase = "draw"
            else:
                if any(a.kind == A.PENG for a in legal):
                    p.passed_peng.add(tile)
                if any(a.kind == A.CHI for a in legal):
                    p.passed_chi.add(tile)
                # Passing hu does not silently waive a later peng/chi decision.
                if any(a.kind == A.HU for a in legal):
                    out.hu_passed.add(action.player)
                    # Separate hu-pass tracking from lower-priority passes below.
                    out.phase = "respond"
                else:
                    out.passed.add(action.player)
        elif action.kind == A.HU:
            from .scoring import base_amount, detect_fan, meld_huxi

            heavenly = state.phase == "opening"
            incoming = (A.CHI, A.PENG, A.WEI, A.STINKY_WEI, A.PAO, A.TI)
            earthly = (
                action.player != out.dealer
                and sum(a.kind == A.DRAW for a in out.history) == 1
                and not any(a.player == action.player and a.kind in incoming for a in out.history)
            )
            structures = self._wins(out, action.player)
            groups = max(
                structures,
                key=lambda gs: (
                    base_amount(sum(meld_huxi(g) for g in gs))
                    * (
                        sum(detect_fan(gs, heavenly=heavenly, earthly=earthly).values())
                        or self.config.ordinary_fan
                    )
                ),
            )
            out.settlement = settle(
                action.player,
                groups,
                ordinary_fan=self.config.ordinary_fan,
                heavenly=heavenly,
                earthly=earthly,
            )
            out.winner = action.player
            out.phase = "terminal"
        else:
            opening = action.source_type == S.INITIAL
            if action.kind == A.CHI:
                consumed = list(action.chi)
                consumed.remove(tile)
                for t in consumed:
                    p.hand.remove(t)
                p.melds.append(Meld(M.CHI, action.chi))
                for pattern in action.bi:
                    for t in pattern:
                        p.hand.remove(t)
                    p.melds.append(Meld(M.CHI, pattern))
            else:
                kind = M(action.kind.value)
                upgrade = (
                    next(
                        (
                            i
                            for i, m in enumerate(p.melds)
                            if m.tiles[0] == tile and m.kind in (M.PENG, M.WEI, M.STINKY_WEI)
                        ),
                        None,
                    )
                    if kind in (M.PAO, M.TI)
                    else None
                )
                count = 4 if kind in (M.PAO, M.TI) else 3
                if upgrade is not None:
                    p.melds[upgrade] = Meld(kind, (tile,) * count)
                else:
                    for _ in range(count if opening else count - 1):
                        p.hand.remove(tile)
                    p.melds.append(Meld(kind, (tile,) * count))
                p.kans.discard(tile)
            if opening:
                p.quad_count += 1
                p.opening_double_ti_pending = p.quad_count >= 2
            else:
                out.pending = None
                out.passed.clear()
                out.hu_passed.clear()
                discard = p.needs_discard_after(action.kind)
                if p.hu_disabled:
                    discard = False
                available = set(p.hand) - (p.kans if self.config.protect_kans else set())
                if discard and not available and action.kind in (A.WEI, A.STINKY_WEI):
                    discard = False
                    if not evaluate_hand(
                        p.hand,
                        p.melds,
                        quad_requires_pair=self.config.quad_requires_pair,
                        protected=p.kans if self.config.protect_kans else (),
                    ):
                        p.hu_disabled = True
                out.turn = action.player if discard else (action.player + 1) % 3
                out.phase = "discard" if discard else "draw"
                # Forced wei/ti must happen before a hu decision on the same draw.
                if action.kind in (A.WEI, A.STINKY_WEI, A.TI) and action.source_type == S.DRAW:
                    out.resume_phase = out.phase
                    out.resume_turn = out.turn
                    out.turn = action.player
                    out.phase = "post_meld"
        return out

    def replay(self, initial, actions):
        state = initial.clone()
        for action in actions:
            state = self.step(state, action)
        return state
