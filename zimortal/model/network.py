"""20 positions x channels -> shared encoder -> policy/value/wait heads."""

import torch
from torch import nn

from .encoding import ACTION_DIM, CHANNELS


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
    def __init__(self, architecture="resnet", width=32):
        super().__init__()
        self.architecture = architecture
        self.width = width
        if architecture == "resnet":
            self.encoder = nn.Sequential(
                nn.Conv1d(CHANNELS, width, 1),
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
                nn.Linear(CHANNELS * 20, 256),
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

    def forward(self, features, actions, mask):
        latent = self.encoder(features)
        embeddings = self.action_encoder(actions)
        logits = (embeddings * self.policy_context(latent).unsqueeze(1)).sum(
            -1
        ) / 8 + self.action_bias(actions).squeeze(-1)
        return logits.masked_fill(~mask, -1e9), self.value(latent).squeeze(-1), self.wait(latent)
