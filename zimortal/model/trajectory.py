"""Visible local group candidates; learned forecasts are not legal completion plans."""

from collections import Counter

import torch
from torch import nn

from zimortal.engine.chi import CHI_PATTERNS

GROUP_PATTERNS = CHI_PATTERNS + tuple((tile,) * size for size in (2, 3, 4) for tile in range(20))
FAN_NAMES = (
    "黑胡",
    "点胡",
    "红胡",
    "三扁",
    "四扁",
    "双漂",
    "十红系列",
    "十八大",
    "十八小",
    "碰碰胡",
    "天胡",
    "地胡",
)


class GroupCandidateEncoder(nn.Module):
    """Encode competing local groups, without claiming a disjoint full-hand solution."""

    def __init__(self, channels):
        super().__init__()
        counts = torch.tensor(
            [[Counter(pattern)[t] for t in range(20)] for pattern in GROUP_PATTERNS],
            dtype=torch.float32,
        )
        self.register_buffer("requirements", counts)
        self.group = nn.Sequential(nn.Linear(channels + 23, 64), nn.SiLU(), nn.Linear(64, 64))
        self.attention = nn.Linear(64, 1)
        self.fusion = nn.Linear(64, 128, bias=False)
        # Compatible warm starts initially preserve the parent's exact decisions.
        nn.init.zeros_(self.fusion.weight)

    def encode_groups(self, features):
        required = self.requirements[None]
        hand = (features[:, 0] * 4).round()[:, None]
        unknown = (4 - (features[:, 0] + features[:, 1]) * 4).round().clamp(0, 4)[:, None]
        missing = (required - hand).clamp_min(0)
        covered = torch.minimum(required, hand).sum(-1) / required.sum(-1)
        possible = (missing <= unknown).all(-1).float()
        local = torch.einsum("bct,gt->bgc", features, self.requirements)
        local = local / self.requirements.sum(-1)[None, :, None]
        scalars = torch.stack((covered, missing.sum(-1) / 4, possible), -1)
        groups = self.group(
            torch.cat((local, required.expand(len(features), -1, -1) / 4, scalars), -1)
        )
        return groups

    def forward(self, features):
        groups = self.encode_groups(features)
        weights = self.attention(groups).softmax(1)
        return self.fusion((weights * groups).sum(1))


def terminal_route_targets(settlement):
    """Actual winner's complete decomposition; future labels never become inputs."""
    tiles = Counter(t for group in settlement.groups for t in group.tiles)
    groups = Counter(tuple(sorted(group.tiles)) for group in settlement.groups)
    fans = set(settlement.fan)
    return (
        [tiles[t] for t in range(20)],
        [groups[p] for p in GROUP_PATTERNS],
        [
            int(any(n.endswith("红") and n != "红胡" for n in fans))
            if name == "十红系列"
            else int(name in fans)
            for name in FAN_NAMES
        ],
    )


def complete_group_plans(scores, capacity, fixed=(), *, beam_size=64, top_k=3, max_quads=None):
    """Budgeted structural targets, not reachable-action or >=15-huxi proofs.

    capacity is the visible optimistic total inventory bound (including fixed
    groups). Fixed groups must be retained exactly; promotion alternatives must
    be supplied separately. Beam pruning means an empty result is unknown.
    No winning-route labels or hidden wall counts are accepted as input.
    """
    import math

    if len(scores) != len(GROUP_PATTERNS) or len(capacity) != 20:
        raise ValueError("invalid plan dimensions")
    if beam_size < 1 or top_k < 1 or any(not math.isfinite(float(x)) for x in scores):
        raise ValueError("invalid plan budget/scores")
    if any(int(n) != n or n < 0 or n > 4 for n in capacity):
        raise ValueError("capacity must be integer copy bounds")
    indices = [GROUP_PATTERNS.index(tuple(sorted(p))) for p in fixed]
    if len(indices) > 7:
        raise ValueError("too many fixed groups")
    requirements = [tuple(Counter(p)[t] for t in range(20)) for p in GROUP_PATTERNS]
    used = tuple(sum(requirements[i][t] for i in indices) for t in range(20))
    if any(n > capacity[t] for t, n in enumerate(used)):
        return {"status": "unknown", "plans": [], "beam_pruned": False}
    # score, chosen groups, inventory, last free index; sorted free additions
    # avoid spending the budget on permutations of the same complete plan.
    beam = [(sum(float(scores[i]) for i in indices), tuple(indices), used, 0)]
    pruned = False
    pair_last = max(j for j, p in enumerate(GROUP_PATTERNS) if len(p) == 2)
    for _ in range(7 - len(indices)):
        expanded = []
        for score, chosen, inventory, first in beam:
            for i in range(first, len(GROUP_PATTERNS)):
                candidate = chosen + (i,)
                pairs = sum(len(GROUP_PATTERNS[j]) == 2 for j in candidate)
                quads = sum(len(GROUP_PATTERNS[j]) == 4 for j in candidate)
                if (
                    (max_quads is not None and quads > max_quads)
                    or pairs > 1
                    or (len(candidate) == 7 and pairs != int(quads > 0))
                ):
                    continue
                # Free additions are sorted. Once the last index passes all
                # pair patterns, a quad prefix without a pair cannot ever
                # complete. Do not let such doomed prefixes consume the beam.
                if quads and not pairs and i > pair_last:
                    continue
                counts = tuple(inventory[t] + requirements[i][t] for t in range(20))
                if any(counts[t] > capacity[t] for t in range(20)):
                    continue
                expanded.append((score + float(scores[i]), candidate, counts, i))
        expanded.sort(key=lambda item: item[0], reverse=True)
        pruned |= len(expanded) > beam_size
        beam = expanded[:beam_size]
        if not beam:
            break
    plans = [
        {"groups": [GROUP_PATTERNS[i] for i in chosen], "tiles": counts, "score": score}
        for score, chosen, counts, _ in beam
        if len(chosen) == 7
        and sum(len(GROUP_PATTERNS[i]) == 2 for i in chosen)
        == int(any(len(GROUP_PATTERNS[i]) == 4 for i in chosen))
    ][:top_k]
    return {
        "status": "structural_candidates" if plans else "unknown",
        "plans": plans,
        "beam_pruned": pruned,
    }


def observation_group_plans(observation, scores, *, extra_fixed=(), **budget):
    """Conservative fixed-meld/kan branch using public copy upper bounds only."""
    own = Counter(observation.hand)
    fixed = [m.tiles for m in observation.players[observation.player].melds]
    fixed += [(tile,) * 3 for tile, count in own.items() if count == 3]
    fixed.extend(extra_fixed)
    if len(fixed) > 7:
        return {"status": "unknown", "plans": [], "beam_pruned": False}
    external = Counter(observation.river)
    for player, public in enumerate(observation.players):
        if player != observation.player:
            external.update(tile for meld in public.melds for tile in meld.tiles)
    # Pending tile remains optimistically available, regardless of who may
    # claim it. The decoder does not certify acquisition legality/probability.
    return complete_group_plans(scores, [4 - external[t] for t in range(20)], fixed, **budget)
