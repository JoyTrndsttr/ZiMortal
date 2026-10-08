"""Frozen visible-state encoder plus an ensemble of candidate cash-Q regressors."""

import torch
from torch import nn

from .encoding import ACTION_DIM, HUXI_CHANNELS, KINDS
from .network import PolicyValueNet


class CashQNet(nn.Module):
    architecture = "cashq"
    feature_version = "huxi"
    auxiliary_version = "legacy"

    def __init__(self, width=32):
        super().__init__()
        self.width = width
        self.parent = PolicyValueNet("resnet", width, "huxi")
        self.parent.requires_grad_(False)
        size = 128 + ACTION_DIM + 3 * HUXI_CHANNELS + 1
        self.q_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(size, 256), nn.SiLU(), nn.Linear(256, 64), nn.SiLU(), nn.Linear(64, 1)
                )
                for _ in range(3)
            ]
        )
        for head in self.q_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        # Inference gate parameters are checkpointed; cash units are /100.
        self.register_buffer("margin_cash", torch.tensor(10.0))
        self.register_buffer("uncertainty_multiplier", torch.tensor(2.0))
        self.register_buffer("enabled", torch.tensor(False))

    def forward_q(self, features, actions, mask):
        with torch.no_grad():
            latent = self.parent.encoder(features)
            original = self.parent._heads(latent, features, actions, mask)
        offset = len(KINDS)
        local = []
        for block in range(3):
            weights = actions[:, :, offset + block * 20 : offset + (block + 1) * 20]
            local.append(torch.einsum("bct,bat->bac", features, weights))
        context = torch.cat(
            (
                latent[:, None].expand(-1, actions.shape[1], -1),
                actions,
                *local,
                original[0].masked_fill(~mask, 0).unsqueeze(-1) / 10,
            ),
            -1,
        )
        residual = torch.stack([head(context).squeeze(-1) for head in self.q_heads], -1)
        return residual + original[1][:, None, None], original

    def forward(self, features, actions, mask):
        ensemble, original = self.forward_q(features, actions, mask)
        cash = ensemble.mean(-1) * 100
        cash = cash.masked_fill(~mask, -1e9)
        baseline = original[0].argmax(-1)
        selected = cash.argmax(-1)
        row = torch.arange(len(features), device=features.device)
        advantage = (ensemble[row, selected] - ensemble[row, baseline]) * 100
        confident = (
            advantage.mean(-1) - self.uncertainty_multiplier * advantage.std(-1) > self.margin_cash
        ) & self.enabled
        scores = torch.where(confident[:, None], cash / 10, original[0]).masked_fill(~mask, -1e9)
        return scores, original[1], original[2]
