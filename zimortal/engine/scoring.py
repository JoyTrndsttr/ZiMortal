"""Hu-xi, additive fan, and zero-sum settlement."""

from collections import Counter
from dataclasses import dataclass

from .tiles import is_big, is_red
from .types import Meld, MeldType


def meld_huxi(meld):
    kind, tiles = meld.kind, meld.tiles
    big = is_big(tiles[0])
    if kind == MeldType.CHI:
        ordered = tuple(sorted(tiles))
        if len({is_big(t) for t in tiles}) == 1 and tuple(t % 10 for t in ordered) in (
            (0, 1, 2),
            (1, 6, 9),
        ):
            return 6 if big else 3
        return 0
    return {
        MeldType.PENG: (1, 3),
        MeldType.WEI: (3, 6),
        MeldType.STINKY_WEI: (3, 6),
        MeldType.KAN: (3, 6),
        MeldType.PAO: (6, 9),
        MeldType.TI: (9, 12),
        MeldType.PAIR: (0, 0),
    }[kind][big]


def base_amount(huxi):
    if huxi < 15:
        raise ValueError("at least 15 hu-xi required")
    return 10 + (huxi - 15) // 3 * 5


def detect_fan(groups, *, heavenly=False, earthly=False):
    tiles = [t for g in groups for t in g.tiles]
    reds = Counter(t for t in tiles if is_red(t))
    red_count = sum(reds.values())
    big = sum(is_big(t) for t in tiles)
    result = {}
    if red_count == 0:
        result["黑胡"] = 5
    if red_count == 1:
        result["点胡"] = 4
    if red_count == 3 and len(reds) == 1:
        result["三扁"] = 3
    if red_count == 4 and len(reds) == 1:
        result["四扁"] = 4
    red_groups = [
        g
        for g in groups
        if g.kind in (MeldType.PENG, MeldType.WEI, MeldType.STINKY_WEI, MeldType.PAO, MeldType.TI)
        and is_red(g.tiles[0])
    ]
    if len(red_groups) == 2 and sum(len(g.tiles) for g in red_groups) == red_count:
        result["双漂"] = 2
    if red_count >= 10:
        result["十红"] = red_count - 8
    if big >= 18:
        result["十八大"] = big - 13
    if len(tiles) - big >= 18:
        result["十八小"] = len(tiles) - big - 13
    if groups and all(g.kind != MeldType.CHI for g in groups):
        result["碰碰胡"] = 5
    if heavenly:
        result["天胡"] = 5
    if earthly:
        result["地胡"] = 5
    return result


@dataclass(frozen=True)
class Settlement:
    winner: int
    huxi: int
    fan: dict[str, int]
    amount_each: int
    payments: tuple[int, int, int]
    groups: tuple[Meld, ...] = ()


def settle(winner, groups, *, ordinary_fan, heavenly=False, earthly=False):
    if winner not in range(3) or ordinary_fan < 1:
        raise ValueError("invalid settlement configuration")
    groups = tuple(groups)
    huxi = sum(meld_huxi(g) for g in groups)
    fan = detect_fan(groups, heavenly=heavenly, earthly=earthly)
    amount = base_amount(huxi) * (sum(fan.values()) or ordinary_fan)
    return Settlement(
        winner,
        huxi,
        fan,
        amount,
        tuple(2 * amount if p == winner else -amount for p in range(3)),
        groups,
    )
