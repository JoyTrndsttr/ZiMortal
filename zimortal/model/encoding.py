"""Fixed tile-position features and structured candidate action features."""

from collections import Counter

import numpy as np

from zimortal.engine import ActionType, MeldType, SourceType, is_red, meld_huxi

KINDS = tuple(ActionType)
MELDS = (MeldType.CHI, MeldType.PENG, MeldType.WEI, MeldType.STINKY_WEI, MeldType.PAO, MeldType.TI)
from zimortal.engine.chi import CHI_PATTERNS

ACTION_DIM = len(KINDS) + 60 + len(CHI_PATTERNS) + 5


def encode_observation(obs, version="legacy"):
    own = Counter(obs.hand)
    public = Counter(obs.river)
    for player in obs.players:
        public.update(t for m in player.melds for t in m.tiles)
    if obs.pending:
        public[obs.pending.tile] += 1
    channels = [
        [own[t] / 4 for t in range(20)],
        [public[t] / 4 for t in range(20)],
        [float(t in obs.passed_chi) for t in range(20)],
        [float(t in obs.passed_peng) for t in range(20)],
    ]
    for offset in range(3):
        player = obs.players[(obs.player + offset) % 3]
        for kind in MELDS:
            counts = Counter(t for m in player.melds if m.kind == kind for t in m.tiles)
            channels.append([counts[t] / 4 for t in range(20)])
    my_melds = obs.players[obs.player].melds
    my_tiles = list(obs.hand) + [t for m in my_melds for t in m.tiles]
    hu = sum(meld_huxi(m) for m in my_melds) + sum(
        6 if t >= 10 else 3 for t, n in own.items() if n == 3
    )
    scalar = [
        hu / 60,
        sum(is_red(t) for t in my_tiles) / 24,
        sum(t >= 10 for t in my_tiles) / 24,
        len(my_tiles) / 24,
        sum(m.kind in (MeldType.PAO, MeldType.TI) for m in my_melds) / 6,
        obs.remaining_tiles / 20,
        float(obs.hu_disabled),
        float(obs.pending is not None and obs.pending.source == SourceType.DRAW),
        float(obs.pending is not None and obs.pending.player == obs.player),
    ]
    channels.extend([[v] * 20 for v in scalar])
    channels.extend(
        [
            [float(t >= 10) for t in range(20)],
            [(t % 10 + 1) / 10 for t in range(20)],
            [float(is_red(t)) for t in range(20)],
            [own[(t + 10) % 20] / 4 for t in range(20)],
            [(own[t - 1] if t % 10 else 0) / 4 for t in range(20)],
            [(own[t + 1] if t % 10 != 9 else 0) / 4 for t in range(20)],
            [sum(own[(t // 10) * 10 + r] for r in (1, 6, 9)) / 12 for t in range(20)],
            [float(obs.pending is not None and obs.pending.tile == t) for t in range(20)],
        ]
    )
    for offset in range(3):
        seat = (obs.player + offset) % 3
        discards = Counter(
            a.tile for a in obs.history if a.player == seat and a.kind == ActionType.DISCARD
        )
        passes = Counter(
            a.tile
            for a in obs.history
            if a.player == seat
            and a.kind == ActionType.PASS
            and not a.forced
            and a.tile is not None
        )
        channels.extend(
            [
                [min(discards[t], 4) / 4 for t in range(20)],
                [min(passes[t], 4) / 4 for t in range(20)],
            ]
        )
    if version == "huxi":
        from zimortal.training.valuation import SCORING_PATTERNS, formed_huxi

        kans = {t for t, n in own.items() if n == 3}
        formed = formed_huxi(obs.hand, my_melds, kans)
        losses = []
        for tile in range(20):
            if own[tile] and tile not in kans:
                hand = list(obs.hand)
                hand.remove(tile)
                losses.append((formed - formed_huxi(hand, my_melds, kans)) / 12)
            else:
                losses.append(0.0)
        channels.extend(
            [
                [15 / 60] * 20,
                [formed / 60] * 20,
                [max(0, 15 - formed) / 15] * 20,
                losses,
                [float(t in kans) for t in range(20)],
            ]
        )
        for pattern, _hu in SCORING_PATTERNS:
            filled = sum(min(own[t], 1) for t in pattern if t not in kans) / 3
            channels.append([filled if t in pattern else 0.0 for t in range(20)])
        channels.append([max(0, 15 - hu) / 15] * 20)
    elif version != "legacy":
        raise ValueError("unknown feature version")
    return np.asarray(channels, dtype=np.float32)


CHANNELS = 45
HUXI_CHANNELS = 55


def encode_action(action, player):
    x = np.zeros(ACTION_DIM, dtype=np.float32)
    x[KINDS.index(action.kind)] = 1
    offset = len(KINDS)
    if action.tile is not None:
        x[offset + action.tile] = 1
    for block, tiles in enumerate((action.chi, tuple(t for g in action.bi for t in g))):
        counts = Counter(tiles)
        for t, n in counts.items():
            x[offset + 20 * (block + 1) + t] = n / 4
    for group in action.bi:
        x[offset + 60 + CHI_PATTERNS.index(tuple(sorted(group)))] += 0.25
    x[-5:] = [
        float(action.forced),
        float(action.source_type == SourceType.DRAW),
        float(action.source_type == SourceType.INITIAL),
        float(action.source_player == player),
        min(len(action.bi), 4) / 4,
    ]
    return x
