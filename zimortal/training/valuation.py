"""Visible-only hu-xi potential, structural completion and settlement valuation."""

from collections import Counter
from functools import lru_cache

import numpy as np

from zimortal.engine import ActionType as A
from zimortal.engine import Meld, evaluate_hand, meld_huxi
from zimortal.engine import MeldType as M
from zimortal.engine.chi import CHI_PATTERNS
from zimortal.engine.scoring import base_amount, detect_fan

SCORING_PATTERNS = tuple(
    (p, 6 if p[0] >= 10 else 3)
    for p in CHI_PATTERNS
    if p in ((0, 1, 2), (1, 6, 9), (10, 11, 12), (11, 16, 19))
)


@lru_cache(maxsize=65536)
def _potential(counts):
    if not any(counts):
        return 0
    tile = next(t for t, n in enumerate(counts) if n)
    rest = list(counts)
    rest[tile] -= 1
    best = _potential(tuple(rest))
    patterns = list(SCORING_PATTERNS) + [((tile,) * 3, 6 if tile >= 10 else 3)]
    for pattern, hu in patterns:
        if tile not in pattern:
            continue
        needed = Counter(pattern)
        if all(counts[t] >= n for t, n in needed.items()):
            remaining = tuple(n - needed[t] for t, n in enumerate(counts))
            best = max(best, hu + _potential(remaining))
    return best


def formed_huxi(hand, melds, kans):
    """Disjoint completed scoring groups; not guaranteed winning hu-xi."""
    counts = Counter(hand)
    secured = sum(meld_huxi(m) for m in melds)
    for t in kans:
        if counts[t] != 3:
            raise ValueError("protected kan must contain three tiles")
        secured += 6 if t >= 10 else 3
        counts[t] -= 3
    return secured + _potential(tuple(counts[t] for t in range(20)))


@lru_cache(maxsize=32768)
def draw_values(hand, melds, kans):
    """Per tile: max structural hu-xi, max legal amount each, additive fan.

    Apply own-draw mandatory wei/ti. Ordinary future fan excludes heavenly/
    earthly bonuses, which require their exact game-history conditions.
    """
    own = Counter(hand)
    used = own + Counter(t for m in melds for t in m.tiles)
    values = []
    for tile in range(20):
        if used[tile] >= 4:
            values.append((-1, 0, 0))
            continue
        h = list(hand)
        ms = list(melds)
        ks = set(kans)
        upgrade = next(
            (
                i
                for i, m in enumerate(ms)
                if m.tiles[0] == tile and m.kind in (M.WEI, M.STINKY_WEI, M.PENG)
            ),
            None,
        )
        if upgrade is not None:
            ms[upgrade] = Meld(M.PAO if ms[upgrade].kind == M.PENG else M.TI, (tile,) * 4)
        elif own[tile] in (2, 3):
            for _ in range(own[tile]):
                h.remove(tile)
            ms.append(Meld(M.WEI if own[tile] == 2 else M.TI, (tile,) * (own[tile] + 1)))
            ks.discard(tile)
        else:
            h.append(tile)
        structures = evaluate_hand(h, ms, quad_requires_pair=True, protected=ks, minimum_huxi=0)
        hu = -1
        amount = fan = 0
        for groups in structures:
            score = sum(meld_huxi(g) for g in groups)
            hu = max(hu, score)
            if score >= 15:
                multiplier = sum(detect_fan(groups).values()) or 1
                payment = base_amount(score) * multiplier
                if payment > amount:
                    amount = payment
                    fan = multiplier
        values.append((hu, amount, fan))
    return tuple(values)


def quality(hand, melds, kans, visible):
    counts = Counter(hand)
    formed = formed_huxi(hand, melds, kans)
    outcomes = draw_values(tuple(sorted(hand)), tuple(melds), tuple(sorted(kans)))
    # Visible missing-copy counts are upper bounds, not probabilities or EV.
    payout = sum(
        max(0, 4 - counts[t] - visible[t]) * amount / 10
        for t, (_hu, amount, _fan) in enumerate(outcomes)
    )
    attainable = max((hu for hu, _, _ in outcomes), default=0)
    below = max(0, 15 - attainable) if attainable >= 0 else max(0, 15 - formed)
    # A threshold deficit must matter even when a hand has a good shape.
    cohesion = sum(n * (n - 1) for n in counts.values())
    for pattern in CHI_PATTERNS:
        required = Counter(pattern)
        filled = sum(min(counts[t], n) for t, n in required.items())
        cohesion += 0.15 * filled * filled
    return payout + 0.3 * formed - 0.4 * below + 0.015 * cohesion


def action_scores(obs):
    melds = obs.players[obs.player].melds
    kans = {t for t, n in Counter(obs.hand).items() if n == 3}
    visible = Counter(obs.river)
    for p in obs.players:
        visible.update(t for m in p.melds for t in m.tiles)
    if obs.pending:
        visible[obs.pending.tile] += 1
    scores = []
    initial_ti = sum(
        a.player == obs.player and a.kind == A.TI and a.source_type.value == "initial"
        for a in obs.history
    )
    intake = any(
        a.player == obs.player
        and a.kind in (A.CHI, A.PENG, A.WEI, A.STINKY_WEI, A.PAO, A.TI)
        and a.source_type.value != "initial"
        for a in obs.history
    )
    waived = initial_ti >= 2 and not intake
    for action in obs.legal_actions:
        if action.kind == A.HU:
            scores.append(1e6)
            continue
        if action.forced:
            scores.append(0.0)
            continue
        if action.kind == A.PASS:
            scores.append(quality(obs.hand, melds, kans, visible) - 0.1)
            continue
        hand = list(obs.hand)
        ms = list(melds)
        if action.kind == A.DISCARD:
            hand.remove(action.tile)
        elif action.kind == A.PENG:
            hand.remove(action.tile)
            hand.remove(action.tile)
            ms.append(Meld(M.PENG, (action.tile,) * 3))
        elif action.kind == A.CHI:
            needed = list(action.chi)
            needed.remove(action.tile)
            for t in needed:
                hand.remove(t)
            ms.append(Meld(M.CHI, action.chi))
            for group in action.bi:
                for t in group:
                    hand.remove(t)
                ms.append(Meld(M.CHI, group))
        public = visible + (Counter(obs.hand) - Counter(hand))
        if action.kind in (A.PENG, A.CHI) and not waived:
            candidates = set(hand) - kans
            choices = []
            for t in candidates:
                post = list(hand)
                post.remove(t)
                choices.append(quality(post, ms, kans, public + Counter({t: 1})))
            score = max(choices, default=-100.0)
        else:
            score = quality(hand, ms, kans, public)
        scores.append(score)
    return np.asarray(scores, np.float32)
