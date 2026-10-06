"""Stable tile types; physical copies are interchangeable."""

import random

Tile = int
SMALL_NAMES = "一二三四五六七八九十"
BIG_NAMES = "壹贰叁肆伍陆柒捌玖拾"


def validate(tile: Tile) -> Tile:
    if type(tile) is not int or not 0 <= tile < 20:
        raise ValueError("tile must be an integer in [0, 20)")
    return tile


def is_big(tile: Tile) -> bool:
    return validate(tile) >= 10


def rank(tile: Tile) -> int:
    return validate(tile) % 10 + 1


def is_red(tile: Tile) -> bool:
    return rank(tile) in (2, 7, 10)


def tile_name(tile: Tile) -> str:
    return (BIG_NAMES if is_big(tile) else SMALL_NAMES)[rank(tile) - 1]


def make_deck(seed: int | None = None) -> list[Tile]:
    deck = [t for t in range(20) for _ in range(4)]
    random.Random(seed).shuffle(deck)
    return deck
