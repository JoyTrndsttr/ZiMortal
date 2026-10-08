"""20 positions x channels -> shared encoder -> policy/value/wait heads."""

import torch
from torch import nn

from .encoding import ACTION_DIM, CHANNELS, HUXI_CHANNELS


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=1),
            nn.GroupNorm(4, width),
            nn.SiLU(),
            nn.Conv1d(width, width, 3, padding=1),
            nn.GroupNorm(4, width),
        )

    def forward(self, x):
        return torch.nn.functional.silu(x + self.net(x))


class PolicyValueNet(nn.Module):
    def __init__(
        self, architecture="resnet", width=32, feature_version="legacy", auxiliary_version="legacy"
    ):
        super().__init__()
        self.architecture = architecture
        self.width = width
        self.feature_version = feature_version
        self.auxiliary_version = auxiliary_version
        if auxiliary_version not in ("legacy", "boundary", "planning"):
            raise ValueError("unknown auxiliary version")
        if auxiliary_version in ("boundary", "planning") and feature_version != "huxi":
            raise ValueError("boundary heads require huxi features")
        self.input_channels = HUXI_CHANNELS if feature_version == "huxi" else CHANNELS
        if architecture == "resnet":
            self.encoder = nn.Sequential(
                nn.Conv1d(self.input_channels, width, 1),
                nn.SiLU(),
                ResidualBlock(width),
                ResidualBlock(width),
                nn.Flatten(),
                nn.Linear(width * 20, 128),
                nn.SiLU(),
            )
        elif architecture == "mlp":
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.input_channels * 20, 256),
                nn.SiLU(),
                nn.Linear(256, 128),
                nn.SiLU(),
            )
        else:
            raise ValueError("unknown architecture")
        self.policy_context = nn.Linear(128, 64)
        self.action_encoder = nn.Sequential(nn.Linear(ACTION_DIM, 64), nn.SiLU(), nn.Linear(64, 64))
        self.action_bias = nn.Linear(ACTION_DIM, 1)
        self.value = nn.Sequential(nn.Linear(128, 32), nn.SiLU(), nn.Linear(32, 1))
        self.wait = nn.Linear(128, 20)
        self.huxi = nn.Linear(128, 21) if feature_version == "huxi" else None
        self.outcome = (
            nn.Sequential(
                nn.Linear(128 + self.input_channels * 2, 128), nn.SiLU(), nn.Linear(128, 4)
            )
            if auxiliary_version in ("boundary", "planning")
            else None
        )
        self.belief = nn.Linear(128, 100) if auxiliary_version == "planning" else None
        self.distance = nn.Linear(128, 16) if auxiliary_version == "planning" else None
        if self.belief is not None:
            nn.init.zeros_(self.belief.weight)
            nn.init.zeros_(self.belief.bias)

    def _heads(self, latent, features, actions, mask):
        embeddings = self.action_encoder(actions)
        logits = (embeddings * self.policy_context(latent).unsqueeze(1)).sum(
            -1
        ) / 8 + self.action_bias(actions).squeeze(-1)
        hu = self.huxi(latent) if self.huxi is not None else None
        eligibility = amount = fan = None
        if self.outcome is not None:
            local = features.transpose(1, 2)
            opposite = torch.cat((local[:, 10:], local[:, :10]), dim=1)
            outcomes = self.outcome(
                torch.cat((latent[:, None].expand(-1, 20, -1), local, opposite), -1)
            )
            hu = torch.cat((hu[:, :1], outcomes[:, :, 0]), dim=1)
            eligibility, amount, fan = (outcomes[:, :, i] for i in (1, 2, 3))
        return (
            logits.masked_fill(~mask, -1e9),
            self.value(latent).squeeze(-1),
            self.wait(latent),
            hu,
            eligibility,
            amount,
            fan,
        )

    def forward_aux(self, features, actions, mask):
        return self._heads(self.encoder(features), features, actions, mask)

    def forward_planning(self, features, actions, mask):
        if self.belief is None:
            raise ValueError("planning heads not enabled")
        latent = self.encoder(features)
        known = ((features[:, 0] + features[:, 1]) * 4).round()
        upper = (4 - known).clamp(0, 4)
        # Learn a residual over the physically valid uniform-allocation prior.
        # This uses only visible counts and remaining wall length, never labels.
        counts = torch.arange(5, device=features.device)[None, None]
        pool = upper.sum(-1)[:, None, None]
        wall = (features[:, 27, 0] * 20).round()[:, None, None]

        def log_comb(n, k):
            score = torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
            return score.masked_fill((k < 0) | (k > n), -1e9)

        prior = (
            log_comb(upper[:, :, None], counts)
            + log_comb(pool - upper[:, :, None], wall - counts)
            - log_comb(pool, wall)
        )
        belief = self.belief(latent).reshape(-1, 20, 5) + prior
        impossible = torch.arange(5, device=features.device)[None, None] > upper[:, :, None]
        belief = belief.masked_fill(impossible, -1e9)
        return self._heads(latent, features, actions, mask), belief, self.distance(latent)

    def forward_all(self, features, actions, mask):
        return self.forward_aux(features, actions, mask)[:4]

    def forward(self, features, actions, mask):
        return self.forward_all(features, actions, mask)[:3]
