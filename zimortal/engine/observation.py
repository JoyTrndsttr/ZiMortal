"""Player-visible information only, with no reference to mutable full state."""

from dataclasses import dataclass

from .types import Action, Meld


@dataclass(frozen=True)
class PublicPlayer:
    hand_count: int
    melds: tuple[Meld, ...]


@dataclass(frozen=True)
class Observation:
    player: int
    hand: tuple[int, ...]
    hu_disabled: bool
    players: tuple[PublicPlayer, ...]
    remaining_tiles: int
    river: tuple[int, ...]
    history: tuple[Action, ...]
    pending: object
    turn: int
    decision_player: int | None
    phase: str
    passed_peng: frozenset[int]
    passed_chi: frozenset[int]
    legal_actions: tuple[Action, ...]


def observe(engine, state, player):
    if player not in range(3):
        raise ValueError("invalid player")
    p = state.players[player]
    # One revealed tile identifies same-type wei/ti without exposing any hand.
    # The rules do not prescribe visibility of ordinary wei; tile type is
    # public here per the requested observation/strategy contract.
    return Observation(
        player,
        tuple(sorted(p.hand)),
        p.hu_disabled,
        tuple(PublicPlayer(len(q.hand), tuple(q.melds)) for q in state.players),
        len(state.deck),
        tuple(state.river),
        tuple(state.history),
        state.pending,
        state.turn,
        engine.decision_player(state),
        state.phase,
        frozenset(p.passed_peng),
        frozenset(p.passed_chi),
        engine.legal_actions(state, player),
    )
