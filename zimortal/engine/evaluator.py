"""Memoized exact-cover hand decomposition, independent of game transitions."""

from collections import Counter
from functools import lru_cache

from .chi import CHI_PATTERNS
from .scoring import meld_huxi
from .tiles import validate
from .types import Meld, MeldType


@lru_cache(maxsize=65536)
def _decompose(counts, groups_left, pair_left):
    if sum(counts) != groups_left * 3 + pair_left * 2:
        return ()
    if not any(counts):
        return ((),) if groups_left == pair_left == 0 else ()
    tile = next(t for t, n in enumerate(counts) if n)
    choices = []
    if pair_left and counts[tile] >= 2:
        choices.append(Meld(MeldType.PAIR, (tile,) * 2))
    if groups_left:
        if counts[tile] >= 3:
            choices.append(Meld(MeldType.KAN, (tile,) * 3))
        choices.extend(Meld(MeldType.CHI, p) for p in CHI_PATTERNS if tile in p)
    result = []
    for group in choices:
        needed = Counter(group.tiles)
        if any(counts[t] < n for t, n in needed.items()):
            continue
        remaining = tuple(n - needed[t] for t, n in enumerate(counts))
        pair = group.kind == MeldType.PAIR
        for tail in _decompose(remaining, groups_left - (not pair), pair_left - pair):
            result.append((group,) + tail)
    return tuple(result)


def evaluate_hand(hand, melds=(), *, quad_requires_pair, protected=()):
    """Return all >=15 hu-xi structures under the explicitly selected quad rule.

    The specification's six-group rule is literal; multiple-quad variants
    must be clarified before using a different structural rule.
    """
    melds = tuple(melds)
    counts = Counter(hand)
    for tile in counts:
        validate(tile)
    total = counts + Counter(t for meld in melds for t in meld.tiles)
    if any(n > 4 for n in total.values()):
        raise ValueError("invalid hand")
    fixed = list(melds)
    for t in protected:
        if counts[t] != 3:
            raise ValueError("protected kan must contain three tiles")
        fixed.append(Meld(MeldType.KAN, (t,) * 3))
        counts[t] -= 3
    if any(g.kind == MeldType.PAIR for g in fixed):
        raise ValueError("pairs must be formed by the evaluator, not exposed melds")
    quad = any(g.kind in (MeldType.PAO, MeldType.TI) for g in fixed)
    pair = int(quad and quad_requires_pair)
    target = 6 if pair else 7
    if target < len(fixed):
        return ()
    structures = _decompose(tuple(counts[t] for t in range(20)), target - len(fixed), pair)
    return tuple(
        tuple(fixed) + s for s in structures if sum(meld_huxi(g) for g in (*fixed, *s)) >= 15
    )
