"""Frozen anchor with an explicit per-game budget; counts actual decisions only."""

import numpy as np
import torch

from .mortal_extraction import ExtractionPolicy


class DeviationPolicy(ExtractionPolicy):
    def __init__(self, q, parent, behavior, budget=None):
        if budget not in (None, 0, 1, 2):
            raise ValueError("predeclared deviation budgets only")
        super().__init__(q, parent, behavior, {"mode": "anchor", "lambda": .3})
        self.budget = budget
        self.counts = {}
        self.context = None

    def prepare(self, specs):
        keys = [(int(s[0]), int(s[2])) for s in specs]
        if len(set(keys)) != len(keys):
            raise ValueError("a game may appear only once per inference batch")
        self.context = keys

    def forward(self, features, actions, mask):
        if self.context is None or len(self.context) != len(features):
            raise RuntimeError("per-game decision context required")
        scores, value, auxiliary = super().forward(features, actions, mask)
        self.proposals = scores.detach().cpu().numpy()
        self.before = [self.counts.get(k, 0) for k in self.context]
        q, p, parent, _keep, _ = self.last
        available = np.array([self.budget is None or n < self.budget for n in self.before])
        self.available = available
        available_t = torch.as_tensor(available, device=mask.device)
        parent_t = torch.as_tensor(parent, device=mask.device)
        retained = torch.zeros_like(mask).scatter_(1, parent_t[:, None], True)
        actual_keep = torch.where(available_t[:, None], mask, retained)
        scores = scores.masked_fill(~actual_keep, -torch.inf)
        self.last = (q, p, parent, actual_keep.cpu().numpy(), scores.detach().cpu().numpy())
        return scores, value, auxiliary

    def record(self, obs, index, chosen, spec, step):
        key = (int(spec[0]), int(spec[2]))
        if self.context[index] != key or self.counts.get(key, 0) != self.before[index]:
            raise RuntimeError("decision context/counter mismatch")
        r = super().record(obs, index, chosen, spec, step)
        n = len(obs.legal_actions)
        proposal = int(self.proposals[index, :n].argmax())
        ranked = np.sort(self.proposals[index, :n])
        if (self.available[index] and chosen != proposal) or (not self.available[index] and chosen != r["parent"]):
            raise RuntimeError("chosen action violates deviation budget")
        self.counts[key] = self.before[index] + int(r["changed"])
        r.update(
            deviation_count_before=self.before[index], deviation_count_after=self.counts[key],
            deviation_budget=self.budget, unlimited_anchor_choice=proposal,
            unlimited_anchor_margin=float(ranked[-1] - ranked[-2]) if n > 1 else None,
            blocked_by_budget=not bool(self.available[index]) and proposal != r["parent"],
        )
        return r
