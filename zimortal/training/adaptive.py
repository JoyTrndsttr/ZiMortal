"""Common-particle adaptive cash rollout with simultaneous sequential bounds."""

import math
from dataclasses import dataclass

import numpy as np

from zimortal.belief.conditional import HistorySampler
from zimortal.engine import ActionType as A
from zimortal.engine import is_red, meld_huxi
from zimortal.engine.scoring import base_amount

from .rollout import teacher


@dataclass(frozen=True)
class AdaptiveConfig:
    minimum: int = 128
    maximum: int = 1048576
    chunk: int = 32
    alpha: float = 0.05
    tie_cash: float = 1.0
    scope: str = "top_set"

    def __post_init__(self):
        if self.minimum < 2 or self.maximum < 2 * self.minimum or self.chunk < 2:
            raise ValueError("require two looks and at least two particles per chunk")
        if self.minimum & (self.minimum - 1) or self.maximum & (self.maximum - 1):
            raise ValueError("look sizes must be powers of two")
        if self.chunk & (self.chunk - 1) or self.chunk > self.minimum:
            raise ValueError("chunk must be a power of two no larger than minimum")
        if not 0 < self.alpha < 1 or self.tie_cash <= 0:
            raise ValueError("invalid confidence level or equivalence margin")
        if self.scope not in ("top_set", "full_order"):
            raise ValueError("unknown certification scope")


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
        scenarios = []
        if not any(len(m.tiles) == 4 for m in ms) and len(ms) <= 7:
            scenarios.append((sum(meld_huxi(m) for m in ms) + (7 - len(ms)) * 6, 21))
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
            scenarios.append(
                (potential + (6 - len(ms)) * 12, maximum_sizes + (6 - len(ms)) * 4 + 2)
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


def empirical_bernstein(mean, variance, n, low, high, delta):
    """Two-sided Maurer-Pontil bound, union of the two one-sided bounds."""
    # HU support endpoints can originate from lossless int16 cash evidence.
    # Promote BEFORE subtracting or multiplying, not after an overflow.
    mean, variance, low, high, delta = map(float, (mean, variance, low, high, delta))
    n = int(n)
    if n < 2 or not 0 < delta < 1 or high < low:
        raise ValueError("invalid bound inputs")
    log = math.log(4 / delta)
    radius = math.sqrt(2 * max(0, variance) * log / n) + 7 * (high - low) * log / (3 * (n - 1))
    return max(low, mean - radius), min(high, mean + radius)


def relations(outcomes, obs, config, look):
    values = np.asarray(outcomes)
    if values.ndim != 2 or len(values) < 2 or values.shape[1] != len(obs.legal_actions):
        raise ValueError("invalid paired outcomes")
    low, high = payoff_bounds(obs)
    if not np.isfinite(values).all() or np.any(values < low) or np.any(values > high):
        raise ValueError("rollout payments violate rule-derived bounds")
    actions = values.shape[1]
    pairs = actions * (actions - 1) // 2
    # Sum_{look>=0} 1/((look+1)(look+2)) = 1. Protect every pair and look.
    delta = config.alpha / ((look + 1) * (look + 2) * max(1, pairs))
    intervals = np.zeros((actions, actions, 2), np.float64)
    codes = np.zeros((actions, actions), np.int8)
    np.fill_diagonal(codes, 2)
    for i in range(actions):
        for j in range(i):
            difference = values[:, i].astype(np.float64) - values[:, j]
            dlow, dhigh = low - high, high - low
            # A HU payment depends only on the acting player's visible hand and
            # public heavenly/earthly conditions, so it is exactly known.
            if obs.legal_actions[i].kind == A.HU:
                if not np.all(values[:, i] == values[0, i]):
                    raise ValueError("same observation has inconsistent HU payments")
                dlow, dhigh = values[0, i] - high, values[0, i] - low
            elif obs.legal_actions[j].kind == A.HU:
                if not np.all(values[:, j] == values[0, j]):
                    raise ValueError("same observation has inconsistent HU payments")
                dlow, dhigh = low - values[0, j], high - values[0, j]
            left, right = empirical_bernstein(
                float(difference.mean()),
                float(difference.var(ddof=1)),
                len(values),
                dlow,
                dhigh,
                delta,
            )
            intervals[i, j], intervals[j, i] = (left, right), (-right, -left)
            code = (
                2
                if left >= -config.tie_cash and right <= config.tie_cash
                else 1
                if left > 0
                else -1
                if right < 0
                else 0
            )
            codes[i, j], codes[j, i] = code, code if code in (0, 2) else -code
    optimal = [i for i in range(actions) if not np.any(codes[:, i] == 1)]
    resolved = bool(optimal) and all(codes[i, j] == 2 for i in optimal for j in optimal)
    resolved &= all(codes[i, j] == 1 for i in optimal for j in range(actions) if j not in optimal)
    if config.scope == "full_order":
        resolved &= not np.any(codes == 0)
    return {
        "n": len(values),
        "look": look,
        "delta_per_pair": delta,
        "payoff_bounds_cash": [low, high],
        "intervals": intervals,
        "relations": codes,
        "optimal": optimal,
        "resolved": bool(resolved),
        "scope": config.scope,
    }


def adaptive_rollout(
    obs,
    model,
    config=None,
    *,
    seed=0,
    reference=0,
    initial=None,
    checkpoint=None,
    max_attempts=2000,
    new_look_limit=None,
):
    """Fresh, independent particle blocks; no recycling screening particles.

    ``checkpoint`` persists each entire scheduled look. Interrupted partial
    looks are regenerated from their deterministic particle offsets.
    """
    config = config or AdaptiveConfig()
    sampler = HistorySampler(obs)
    restored = np.asarray(initial if initial is not None else [])
    if restored.size and (
        restored.ndim != 2
        or restored.shape[1] != len(obs.legal_actions)
        or not np.all(np.isfinite(restored))
        or np.any(restored != np.floor(restored))
        or np.any(restored < -32768)
        or np.any(restored > 32767)
    ):
        raise ValueError("invalid lossless particle evidence")
    values = restored.astype(np.int16)
    if not values.size:
        values = np.empty((0, len(obs.legal_actions)), np.int16)
    if len(values) and (len(values) < config.minimum or len(values) & (len(values) - 1)):
        raise ValueError("resume requires a complete scheduled look")
    trace, previous, result = [], None, None
    new_looks = 0
    look = 0
    target = config.minimum
    while target <= config.maximum:
        if len(values) >= target:
            result = relations(values[:target], obs, config, look)
        else:
            new_looks += 1
            blocks = [values]
            count = len(values)
            while count < target:
                size = min(config.chunk, target - count)
                raw = teacher(
                    obs,
                    model=model,
                    rollouts=size,
                    seed=seed,
                    particle_offset=count,
                    world_sampler=sampler,
                    batched=True,
                    keep_outcomes=True,
                    reference_index=reference,
                    max_attempts=max_attempts,
                )
                block = np.asarray(raw.outcomes_cash, np.int64)
                if np.any(block < -32768) or np.any(block > 32767):
                    raise ValueError("cash payment does not fit lossless int16 evidence")
                blocks.append(block.astype(np.int16))
                count += size
            values = np.concatenate(blocks)
            result = relations(values, obs, config, look)
        signature = (
            result["relations"].tobytes()
            if config.scope == "full_order"
            else tuple(result["optimal"])
        )
        qualified = result["resolved"] and previous == signature
        trace.append({k: v for k, v in result.items() if k not in ("intervals", "relations")})
        if checkpoint is not None and len(values) == target:
            checkpoint(values, result, trace, qualified)
        if qualified:
            return values[:target], result, trace, "qualified"
        previous = signature if result["resolved"] else None
        if new_look_limit is not None and new_looks >= new_look_limit and target < config.maximum:
            return values, result, trace, "pending"
        look += 1
        target *= 2
    return values, result, trace, "budget_exhausted"
