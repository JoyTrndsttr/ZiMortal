"""Presentation-only hand arrangement; never a winning decomposition or policy."""

from collections import Counter
from itertools import combinations
from math import ceil


def _affinity(left, right):
    tiles = left + right
    ranks = {t % 10 for t in tiles}
    suits = {t >= 10 for t in tiles}
    if len(suits) == 1 and ranks == {0, 1, 2}:
        return 150
    if len(suits) == 1 and ranks == {1, 6, 9}:
        return 145
    if len(ranks) == 1:
        return 100  # Same rank, including small/big combinations.
    if len(suits) == 1:
        if ranks <= {0, 1, 2}:
            return 90
        if ranks <= {1, 6, 9}:
            return 85
        if max(ranks) - min(ranks) <= 2:
            return 65
    if max(ranks) - min(ranks) <= 1:
        return 40
    return -10 * (max(ranks) - min(ranks))


def arrange_hand(hand, kans=()):
    """Keep identical copies together, then merge related columns up to four.

    Aim for seven opening columns; shorter hands use proportionally fewer.
    Locked kans remain together. Low-affinity leftovers share a column only
    when needed to reach the target. Columns are visual groups, not melds.
    """
    counts = Counter(hand)
    columns = [[t] * n for t, n in sorted(counts.items())]
    target = min(7, max(ceil(len(hand) / 4), ceil(len(hand) / 3)))
    locked = set(kans)
    while len(columns) > target:
        choices = []
        for size in (2, 3):
            for indices in combinations(range(len(columns)), size):
                merged = [t for i in indices for t in columns[i]]
                if len(merged) > 4 or any(t in locked for t in merged):
                    continue
                score = _affinity(merged, [])
                # Three-column merges are reserved for complete special patterns.
                if size == 3 and score < 145:
                    continue
                choices.append((score, tuple(-i for i in indices), indices, merged))
        if not choices:
            break
        _, _, indices, merged = max(choices)
        columns[indices[0]] = sorted(merged, key=lambda t: (t % 10, t >= 10))
        for i in reversed(indices[1:]):
            columns.pop(i)
    return sorted(columns, key=lambda col: (min(t % 10 for t in col), col[0] >= 10, col))
