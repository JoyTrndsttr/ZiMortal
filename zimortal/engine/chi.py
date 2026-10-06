"""Chi patterns and exhaustive, memoized mandatory-bi completion."""

from collections import Counter
from functools import lru_cache

CHI_PATTERNS = tuple(
    sorted(
        set(
            [tuple(base + r + i for i in range(3)) for base in (0, 10) for r in range(8)]
            + [(base + 1, base + 6, base + 9) for base in (0, 10)]
            + [(r, r, r + 10) for r in range(10)]
            + [(r, r + 10, r + 10) for r in range(10)]
        )
    )
)


def enumerate_chi_with_required_bi(hand, tile, *, protected=()):
    """Require every remaining copy of the claimed tile to participate in bi.

    This interpretation is explicit at the engine configuration boundary.
    Protected opening kans cannot be dismantled.
    """
    counts = Counter(hand)
    for t in protected:
        counts[t] = 0
    patterns = [p for p in CHI_PATTERNS if tile in p]

    def consume(state, pattern):
        needed = Counter(pattern)
        if any(state[t] < n for t, n in needed.items()):
            return None
        return tuple(state[t] - needed[t] for t in range(20))

    @lru_cache(None)
    def complete(state):
        if not state[tile]:
            return ((),)
        results = set()
        for pattern in patterns:
            remaining = consume(state, pattern)
            if remaining is not None:
                for tail in complete(remaining):
                    results.add(tuple(sorted((pattern,) + tail)))
        return tuple(sorted(results))

    initial = tuple(counts[t] for t in range(20))
    result = []
    for pattern in patterns:
        needed = list(pattern)
        needed.remove(tile)
        remaining = consume(initial, needed)
        if remaining is not None:
            result.extend((pattern, bi) for bi in complete(remaining))
    return tuple(result)
