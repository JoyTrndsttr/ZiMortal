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

    def forward(self, features):
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
