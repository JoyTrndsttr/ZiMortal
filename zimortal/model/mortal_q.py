"""Independent visible ResNet Brain and structured legal-action dueling Q."""

import numpy as np
import torch
from torch import nn

from zimortal.engine.observation import Observation

from .encoding import ACTION_DIM, HUXI_CHANNELS, encode_action, encode_observation
from .network import ResidualBlock


class MortalQNet(nn.Module):
    architecture = "mortal_q"
    feature_version = "huxi"
    auxiliary_version = "mortal_mc_v1"

    def __init__(self, width=32, latent=256, blocks=3):
        super().__init__()
        if width < 4 or width % 4 or min(latent, blocks) < 1:
            raise ValueError("invalid Brain dimensions")
        self.width, self.latent, self.blocks = width, latent, blocks
        self.brain = nn.Sequential(
            nn.Conv1d(HUXI_CHANNELS, width, 1),
            nn.SiLU(),
            *(ResidualBlock(width) for _ in range(blocks)),
            nn.Flatten(),
            nn.Linear(width * 20, latent),
            nn.SiLU(),
        )
        self.action_encoder = nn.Sequential(nn.Linear(ACTION_DIM, 64), nn.SiLU())
        self.advantage = nn.Sequential(nn.Linear(latent + 64, 128), nn.SiLU(), nn.Linear(128, 1))
        self.value = nn.Sequential(nn.Linear(latent, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward_q(self, features, actions, mask):
        if features.ndim != 3 or features.shape[1:] != (55, 20):
            raise ValueError("Brain requires 55x20 visible features")
        if actions.ndim != 3 or actions.shape[-1] != ACTION_DIM or mask.shape != actions.shape[:2]:
            raise ValueError("invalid structured action dimensions")
        if mask.dtype != torch.bool or not mask.any(-1).all():
            raise ValueError("every state requires a legal action")
        z = self.brain(features)
        a = self.action_encoder(actions)
        advantage = self.advantage(
            torch.cat((z[:, None].expand(-1, a.shape[1], -1), a), -1)
        ).squeeze(-1)
        mean = advantage.masked_fill(~mask, 0).sum(-1) / mask.sum(-1)
        value = self.value(z).squeeze(-1)
        q = value[:, None] + advantage - mean[:, None]
        return q.masked_fill(~mask, -torch.inf), value, {}

    def forward(self, features, actions, mask):
        return self.forward_q(features, actions, mask)

    def observation_q(self, observation, device="cpu"):
        if not isinstance(observation, Observation):
            raise TypeError("only Observation is accepted")
        features = torch.as_tensor(encode_observation(observation, "huxi")[None], device=device)
        actions = torch.as_tensor(
            np.stack([encode_action(a, observation.player) for a in observation.legal_actions])[
                None
            ],
            device=device,
        )
        return self.forward_q(
            features, actions, torch.ones(actions.shape[:2], dtype=torch.bool, device=device)
        )
