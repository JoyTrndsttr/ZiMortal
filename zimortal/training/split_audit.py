"""Visible, static high-huxi plans; these diagnostics are not action EV."""

from collections import Counter
from functools import cache, lru_cache

from zimortal.engine.scoring import meld_huxi
from zimortal.engine.types import Meld, MeldType

PATTERNS = ((0, 1, 2), (1, 6, 9), (10, 11, 12), (11, 16, 19))


@lru_cache(maxsize=8192)
def plan(counts):
    # Opening kans cannot participate in chi. Only the four scoring chi
    # patterns are considered; this is NOT complete hand evaluation/shanten.
    available = tuple(0 if n >= 3 else n for n in counts)

    @cache
    def solve(left, index):
        if index == len(PATTERNS):
            return 0, ()
        pattern = PATTERNS[index]
        best = solve(left, index + 1)
        remaining = list(left)
        for copies in range(1, min(left[t] for t in pattern) + 1):
            for t in pattern:
                remaining[t] -= 1
            points, groups = solve(tuple(remaining), index + 1)
            candidate = (
                points + copies * (6 if pattern[0] >= 10 else 3),
                (pattern,) * copies + groups,
            )
            if candidate[0] > best[0]:
                best = candidate
        return best

    points, groups = solve(available, 0)
    return points, groups


def diagnostics(hand, action):
    counts = Counter(hand)
    before = tuple(counts[t] for t in range(20))
    prior, groups = plan(before)
    melds = []
    if action.kind.value == "peng":
        counts[action.tile] -= 2
        melds = [Meld(MeldType.PENG, (action.tile,) * 3)]
    elif action.kind.value == "chi":
        melds = [Meld(MeldType.CHI, g) for g in (action.chi, *action.bi)]
        for meld in melds:
            counts.subtract(meld.tiles)
        counts[action.tile] += 1  # Incoming public tile was not in hand.
    if min(counts.values(), default=0) < 0:
        raise ValueError("diagnostics received an unavailable action")
    after = tuple(counts[t] for t in range(20))
    future, remaining = plan(after)
    formed = Counter(tuple(sorted(m.tiles)) for m in melds if m.kind == MeldType.CHI)
    broken = []
    for pattern in PATTERNS:
        old = min(0 if before[t] >= 3 else before[t] for t in pattern)
        new = min(0 if after[t] >= 3 else after[t] for t in pattern)
        if old > new + formed[pattern]:
            broken.append(list(pattern))
    meld_points = sum(meld_huxi(m) for m in melds)
    return {
        "before_pattern_huxi": prior,
        "after_pattern_huxi": future,
        "new_meld_huxi": meld_points,
        "static_huxi_delta": future + meld_points - prior,
        "before_plan": [list(g) for g in groups],
        "after_plan": [list(g) for g in remaining],
        "broken_patterns": broken,
        "mixed_pair": action.kind.value == "chi" and len(set(action.chi)) == 2,
        "board": action.kind.value == "chi"
        and action.tile < 10
        and sum(t >= 10 for t in action.chi) == 2
        and len(set(action.chi)) == 2,
        "interpretation": "static non-overlapping 123/2710 potential; excludes locked kans; not EV",
    }
