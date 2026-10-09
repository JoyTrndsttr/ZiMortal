"""Private rollout engine: reuse legality within an immutable state visit.

Only use with rollout.py, whose transitions always clone the input state.
The public RuleEngine remains uncached and supports caller mutations.
"""

from functools import lru_cache

from zimortal.engine import RuleEngine
from zimortal.engine.chi import enumerate_chi_with_required_bi
from zimortal.engine.evaluator import evaluate_hand


@lru_cache(maxsize=8192)
def _evaluate_cached(hand, melds, quad_requires_pair, protected, exposed_triplet, minimum_huxi):
    return evaluate_hand(
        hand,
        melds,
        quad_requires_pair=quad_requires_pair,
        protected=protected,
        exposed_triplet=exposed_triplet,
        minimum_huxi=minimum_huxi,
    )


def cached_evaluate_hand(
    hand, melds=(), *, quad_requires_pair, protected=(), exposed_triplet=None, minimum_huxi=15
):
    return _evaluate_cached(
        tuple(sorted(hand)),
        tuple(melds),
        quad_requires_pair,
        tuple(sorted(protected)),
        exposed_triplet,
        minimum_huxi,
    )


@lru_cache(maxsize=8192)
def _chi_cached(hand, tile, protected):
    return enumerate_chi_with_required_bi(hand, tile, protected=protected)


def cached_chi(hand, tile, *, protected=()):
    return _chi_cached(tuple(sorted(hand)), tile, tuple(sorted(protected)))


class RolloutEngine(RuleEngine):
    def __init__(self, config=None):
        super().__init__(config)
        self._visited_state = None
        self._visited_actions = ()

    def legal_actions(self, state, player=None):
        if state is not self._visited_state:
            self._visited_actions = super().legal_actions(state)
            self._visited_state = state
        actions = self._visited_actions
        return actions if player is None else tuple(a for a in actions if a.player == player)


def initialize_worker(parent):
    from zimortal.engine import game

    from . import active, rollout

    active.initialize_worker(parent)
    rollout.RuleEngine = RolloutEngine
    game.evaluate_hand = cached_evaluate_hand
    game.enumerate_chi_with_required_bi = cached_chi
