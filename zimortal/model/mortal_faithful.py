"""Mortal v3-style visible Brain; variable Paohuzi action advantage adapter."""

import torch
from torch import nn

from .encoding import ACTION_DIM, HUXI_CHANNELS
from .mortal_q import MortalQNet


class ChannelAttention(nn.Module):
    """Shared average/max pooled channel MLP, as in Mortal's ResBlock."""

    def __init__(self, channels, ratio=16):
        super().__init__()
        if channels < ratio or ratio < 1:
            raise ValueError("attention requires channels >= ratio >= 1")
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, channels // ratio),
            nn.Mish(inplace=True),
            nn.Linear(channels // ratio, channels),
        )
        for layer in self.shared_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        summaries = torch.stack((x.mean(-1), x.amax(-1)), dim=1)
        gates = self.shared_mlp(summaries).sum(dim=1).sigmoid()
        return x * gates.unsqueeze(-1)


class MortalResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.res_unit = nn.Sequential(
            nn.BatchNorm1d(channels, momentum=0.01, eps=1e-3),
            nn.Mish(inplace=True),
            nn.Conv1d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(channels, momentum=0.01, eps=1e-3),
            nn.Mish(inplace=True),
            nn.Conv1d(channels, channels, 3, padding=1, bias=False),
        )
        self.ca = ChannelAttention(channels)

    def forward(self, x):
        return x + self.ca(self.res_unit(x))


class MortalFaithfulQNet(MortalQNet):
    architecture = "mortal_faithful_v1"
    auxiliary_version = "none"

    def __init__(self, width=128, latent=1024, blocks=12):
        nn.Module.__init__(self)
        if width < 16 or latent != 1024 or blocks < 1:
            raise ValueError("faithful Brain requires width>=16, latent1024 and positive blocks")
        self.width, self.latent, self.blocks = width, latent, blocks
        self.brain = nn.Sequential(
            nn.Conv1d(HUXI_CHANNELS, width, 3, padding=1, bias=False),
            *(MortalResidualBlock(width) for _ in range(blocks)),
            nn.BatchNorm1d(width, momentum=0.01, eps=1e-3),
            nn.Mish(inplace=True),
            nn.Conv1d(width, 32, 3, padding=1),
            nn.Mish(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 20, latent),
            nn.Mish(inplace=True),
        )
        self.action_encoder = nn.Sequential(nn.Linear(ACTION_DIM, 64), nn.Mish(inplace=True))
        self.advantage = nn.Sequential(
            nn.Linear(latent + 64, 256), nn.Mish(inplace=True), nn.Linear(256, 1)
        )
        self.value = nn.Sequential(nn.Linear(latent, 256), nn.Mish(inplace=True), nn.Linear(256, 1))
