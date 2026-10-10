"""Deterministic public-information payoff envelope, conserving quad tile types.

Each free quad needs a distinct type with all four physical copies still live.
The other free groups contribute at most six hu-xi. Existing exposed groups
keep the original optimistic upgrade allowance. No sampled rewards are used.
"""

from functools import lru_cache

from zimortal.engine import ActionType as A
from zimortal.engine import is_red, meld_huxi
from zimortal.engine.scoring import base_amount


@lru_cache(maxsize=4096)
def group_huxi_upper(groups, big, small, big_quads, small_quads):
    """Relax identities, conserving size counts and distinct quad types.

    Ordinary big/small groups score <=6/3; mixed groups score zero.
    Pair tiles and fourth tiles upgrading exposed groups are not subtracted,
    deliberately retaining an optimistic envelope.
    """
    best = 0
    for qb in range(min(groups, big_quads, big // 4) + 1):
        for qs in range(min(groups - qb, small_quads, small // 4) + 1):
            ordinary = groups - qb - qs
            if 4 * (qb + qs) + 3 * ordinary > big + small:
                continue
            for b in range(min(ordinary, (big - 4 * qb) // 3) + 1):
                small_groups = min(ordinary - b, (small - 4 * qs) // 3)
                best = max(best, 12 * qb + 9 * qs + 6 * b + 3 * small_groups)
    return best


def payoff_bounds(obs):
    """Rule-derived envelope including future upgrades of exposed triplets.

    Compare the 7-triplet structure with the 6-group-plus-pair structure.
    Public groups and protected private kans cannot lose their tile colours.
    At most 24 physical reds and 40 tiles of each size exist. Fan cases for
    red counts are exclusive, and big/small bonuses cannot coexist (<36 tiles).
    """
    from collections import Counter

    from zimortal.engine import MeldType as M

    heavenly = not any(a.kind == A.DISCARD for a in obs.history)
    earthly = sum(a.kind == A.DRAW for a in obs.history) <= 1
    bonus = 5 if heavenly or earthly else 0  # mutually exclusive engine events
    caps = []
    for seat, player in enumerate(obs.players):
        ms = player.melds
        fixed = [t for m in ms for t in m.tiles]
        if seat == obs.player:
            fixed += [t for t, n in Counter(obs.hand).items() if n == 3 for _ in range(3)]
        fixed_red = sum(is_red(t) for t in fixed)
        fixed_big = sum(t >= 10 for t in fixed)
        dead = list(obs.river) + [
            t for i, p in enumerate(obs.players) if i != seat for m in p.melds for t in m.tiles
        ]
        available = Counter({t: 4 for t in range(20)})
        available.subtract(dead)
        available.subtract(t for m in ms for t in m.tiles)
        big_supply = sum(max(0, available[t]) for t in range(10, 20))
        small_supply = sum(max(0, available[t]) for t in range(10))
        big_quads = sum(available[t] >= 4 for t in range(10, 20))
        small_quads = sum(available[t] >= 4 for t in range(10))
        scenarios = []
        if not any(len(m.tiles) == 4 for m in ms) and len(ms) <= 7:
            scenarios.append(
                (
                    sum(meld_huxi(m) for m in ms)
                    + group_huxi_upper(7 - len(ms), big_supply, small_supply, 0, 0),
                    21,
                )
            )
        if len(ms) <= 6:
            potential = sum(
                (9 if m.tiles[0] >= 10 else 6)
                if m.kind == M.PENG
                else (12 if m.tiles[0] >= 10 else 9)
                if m.kind in (M.WEI, M.STINKY_WEI)
                else meld_huxi(m)
                for m in ms
            )
            maximum_sizes = sum(
                4 if m.kind in (M.PENG, M.WEI, M.STINKY_WEI) else len(m.tiles) for m in ms
            )
            # A future quad consumes all four copies of a distinct tile type.
            # Permanently dead tiles and our existing public groups cannot be reused.
            free = 6 - len(ms)
            quads = min(free, big_quads + small_quads)
            # Other groups have <=6 hu-xi; each possible quad adds <=6.
            scenarios.append(
                (
                    potential
                    + group_huxi_upper(free, big_supply, small_supply, big_quads, small_quads),
                    maximum_sizes + free * 3 + quads + 2,
                )
            )
        amounts = [0]
        for huxi, size in scenarios:
            if huxi < 15:
                continue
            red_upper = min(size - (len(fixed) - fixed_red), 24 - sum(is_red(t) for t in dead))
            red_fan = max(
                [0]
                + [
                    5
                    if r == 0
                    else 4
                    if r == 1
                    else max(3 if r == 3 else 4 if r == 4 else 2, r - 8 if r >= 10 else 0)
                    for r in range(fixed_red, red_upper + 1)
                ]
            )
            big = min(size - (len(fixed) - fixed_big), 40 - sum(t >= 10 for t in dead))
            small = min(size - fixed_big, 40 - sum(t < 10 for t in dead))
            size_fan = max(big - 13 if big >= 18 else 0, small - 13 if small >= 18 else 0)
            triplets = 0 if any(m.kind == M.CHI for m in ms) else 5
            fan = max(1, red_fan + size_fan + triplets + bonus)
            amounts.append(base_amount(huxi) * fan)
        caps.append(max(amounts))
    if obs.hu_disabled:
        caps[obs.player] = 0
    return -max(caps[p] for p in range(3) if p != obs.player), 2 * caps[obs.player]
