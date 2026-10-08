"""Independent, weighted hidden allocations conditioned on forced history evidence.

Dynamic programming removes locally impossible allocations BEFORE rejection.
Full replay still decides acceptance. No MCMC, particles copied from a true game,
reward-conditioned rejection, or correlated particle reuse.
"""

import random
from collections import Counter
from functools import lru_cache
from math import factorial

from zimortal.engine import ActionType as A
from zimortal.engine import GameState, PlayerState, RuleClarificationRequired, RuleEngine
from zimortal.engine import MeldType as M
from zimortal.engine import SourceType as S

from .sampling import SamplingFailure, observed_draws, reverse_hands


def count_constraints(obs):
    previous, seen = {}, [{} for _ in range(3)]
    opening_quads = [set() for _ in range(3)]
    consumed = [set() for _ in range(3)]
    for i, a in enumerate(obs.history):
        if a.kind == A.TI and a.source_type == S.INITIAL:
            opening_quads[a.player].add(a.tile)
        if a.kind == A.DISCARD:
            consumed[a.player].add(a.tile)
        elif a.kind == A.CHI:
            consumed[a.player].update(a.chi)
            consumed[a.player].update(t for g in a.bi for t in g)
        elif a.kind in (A.PENG, A.WEI, A.STINKY_WEI):
            consumed[a.player].add(a.tile)
        if a.kind in (A.PENG, A.WEI, A.STINKY_WEI, A.PAO, A.TI):
            old = seen[a.player].get(a.tile)
            previous[i] = old if a.kind in (A.PAO, A.TI) and a.source_type != S.INITIAL else None
            seen[a.player][a.tile] = M(a.kind.value)
    draws = observed_draws(obs)
    cursor = len(draws) - 1
    events = [[[] for _ in range(20)] for _ in range(3)]
    for i in range(len(obs.history) - 1, -1, -1):
        a = obs.history[i]
        if a.kind == A.DRAW:
            tile = draws[cursor]
            cursor -= 1
            following = obs.history[i + 1] if i + 1 < len(obs.history) else obs.legal_actions[0]
            mandatory = following.player == a.player and following.kind in (
                A.WEI,
                A.STINKY_WEI,
                A.TI,
            )
            permitted = (
                (2,) if mandatory and following.kind != A.TI else (0, 3) if mandatory else (0, 1)
            )
            events[a.player][tile].append(("draw", permitted))
        elif a.kind == A.DISCARD:
            events[a.player][a.tile].append(("discard", 1))
        elif a.kind == A.CHI:
            # Full bi must leave no same-size offered tile in the hand.
            events[a.player][a.tile].append(("empty", 0))
            used = Counter(a.chi)
            used[a.tile] -= 1
            used.update(t for g in a.bi for t in g)
            for tile, n in used.items():
                events[a.player][tile].append(("add", n))
        elif a.kind in (A.PENG, A.WEI, A.STINKY_WEI, A.PAO, A.TI) and previous[i] is None:
            n = 4 if a.source_type == S.INITIAL else 3 if a.kind in (A.PAO, A.TI) else 2
            # A base peng/wei requires exactly two, a private-kan pao/ti
            # exactly three, an opening ti exactly four BEFORE the meld.
            events[a.player][a.tile].append(("meld", n))
    permitted_counts = []
    opening_completed = any(
        a.kind != A.TI or a.source_type != S.INITIAL for a in (*obs.history, *obs.legal_actions)
    )
    for player in range(3):
        allowed = []
        for tile in range(20):
            values = []
            for root_count in range(5):
                count, valid = root_count, True
                for kind, value in events[player][tile]:
                    if kind == "draw":
                        valid &= count in value
                    elif kind == "empty":
                        valid &= count == 0
                    else:
                        count += value
                        if kind == "discard":
                            valid &= count < 3
                        elif kind == "meld":
                            valid &= count == value
                if count == 4 and opening_completed and tile not in opening_quads[player]:
                    valid = False
                if count == 3 and tile in consumed[player]:
                    valid = False
                if valid and count <= 4:
                    values.append(root_count)
            allowed.append(tuple(values))
        permitted_counts.append(tuple(allowed))
    return permitted_counts


class HistorySampler:
    def __init__(self, obs):
        self.obs = obs
        self.engine = RuleEngine()
        self.draws = observed_draws(obs)
        first = next((a for a in obs.history if a.kind == A.DISCARD), None)
        self.dealer = first.player if first else obs.turn
        self.opponents = [p for p in range(3) if p != obs.player]
        known = Counter(obs.hand) + Counter(obs.river)
        for p in obs.players:
            known.update(t for m in p.melds for t in m.tiles)
        if obs.pending:
            known[obs.pending.tile] += 1
        if any(n > 4 for n in known.values()):
            raise ValueError("invalid visible physical counts")
        pool = tuple(4 - known[t] for t in range(20))
        self.sizes = tuple(obs.players[p].hand_count for p in self.opponents)
        if sum(pool) != obs.remaining_tiles + sum(self.sizes):
            raise ValueError("hidden slot counts disagree")
        allowed = count_constraints(obs)
        self.options = []
        for tile, total in enumerate(pool):
            choices = []
            for first_count in allowed[self.opponents[0]][tile]:
                for second_count in allowed[self.opponents[1]][tile]:
                    deck = total - first_count - second_count
                    if deck >= 0:
                        weight = factorial(total) // (
                            factorial(first_count) * factorial(second_count) * factorial(deck)
                        )
                        choices.append((first_count, second_count, deck, weight))
            self.options.append(choices)

        @lru_cache(None)
        def ways(tile, first_left, second_left):
            if min(first_left, second_left) < 0:
                return 0
            if tile == 20:
                return int(first_left == second_left == 0)
            return sum(
                w * ways(tile + 1, first_left - a, second_left - b)
                for a, b, _, w in self.options[tile]
                if a <= first_left and b <= second_left
            )

        self.ways = ways
        if not ways(0, *self.sizes):
            raise SamplingFailure("forced-history constraints have no feasible allocation")

    def allocation(self, rng):
        hands = [list(self.obs.hand) if p == self.obs.player else [] for p in range(3)]
        deck = []
        left = self.sizes
        for tile in range(20):
            ticket = rng.randrange(self.ways(tile, *left))
            for a, b, d, weight in self.options[tile]:
                mass = weight * self.ways(tile + 1, left[0] - a, left[1] - b)
                if ticket < mass:
                    hands[self.opponents[0]].extend([tile] * a)
                    hands[self.opponents[1]].extend([tile] * b)
                    deck.extend([tile] * d)
                    left = (left[0] - a, left[1] - b)
                    break
                ticket -= mass
            else:
                raise RuntimeError("conditional allocation draw failed")
        rng.shuffle(deck)
        return hands, deck

    def sample(self, seed, max_attempts=2000):
        rng = random.Random(seed)
        for attempt in range(1, max_attempts + 1):
            hands, deck = self.allocation(rng)
            try:
                original = reverse_hands(self.obs, [h.copy() for h in hands], self.draws)
                if any(len(h) != (21 if p == self.dealer else 20) for p, h in enumerate(original)):
                    continue
                initial = GameState(
                    [
                        PlayerState(h, kans={t for t, n in Counter(h).items() if n == 3})
                        for h in original
                    ],
                    deck + list(reversed(self.draws)),
                    dealer=self.dealer,
                    turn=self.dealer,
                    phase="opening",
                )
                initial.validate()
                state = self.engine.replay(initial, self.obs.history)
                state.validate()
                if self.engine.observation(state, self.obs.player) == self.obs:
                    return state, attempt
            except (ValueError, RuleClarificationRequired):
                continue
        raise SamplingFailure(
            f"no fully replay-consistent world in {max_attempts} conditional proposals"
        )
