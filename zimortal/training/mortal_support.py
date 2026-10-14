"""Visible behavior support: score full legal Q before deployment filtering."""

from dataclasses import asdict

import numpy as np
import torch
from torch import nn

from zimortal.model.encoding import KINDS

from .mortal_data import action_family
from .mortal_diagnostics import distribution


def behavior_probability(logits, actions, mask, config):
    kinds = actions[..., : len(KINDS)].argmax(-1)
    temps = torch.full_like(logits[:, 0], config["discard"])
    for name in ("peng", "chi", "hu"):
        index = next(i for i, k in enumerate(KINDS) if k.value == name)
        temps = torch.where(((kinds == index) & mask).any(-1), config[f"{name}/pass"], temps)
    p = (logits / temps[:, None]).masked_fill(~mask, -torch.inf).softmax(-1)
    hu_index = next(i for i, k in enumerate(KINDS) if k.value == "hu")
    pass_index = next(i for i, k in enumerate(KINDS) if k.value == "pass")
    hu, passed = (kinds == hu_index) & mask, (kinds == pass_index) & mask
    override = hu.any(-1) & passed.any(-1)
    special = hu * ((1 - config["hu_pass_probability"]) / hu.sum(-1).clamp_min(1))[:, None]
    special = (
        special + passed * (config["hu_pass_probability"] / passed.sum(-1).clamp_min(1))[:, None]
    )
    return torch.where(override[:, None], special, p)


def support_mask(p, mask, parent, threshold):
    if threshold is None:
        return mask.clone()
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("invalid support threshold")
    keep = mask & (p > threshold)
    if not mask.gather(1, parent[:, None]).all():
        raise ValueError("parent greedy must be legal")
    keep.scatter_(1, parent[:, None], True)
    return keep


class SupportPolicy(nn.Module):
    feature_version = "huxi"

    def __init__(self, q, parent, config, threshold=None):
        super().__init__()
        self.q, self.parent, self.config, self.threshold = q, parent, config, threshold
        self.last = None

    def forward(self, features, actions, mask):
        logits, value, _ = self.parent(features, actions, mask)
        p = behavior_probability(logits, actions, mask, self.config)
        parent = logits.argmax(-1)
        q = self.q.forward_q(features, actions, mask)[0] if self.q is not None else logits
        keep = support_mask(p, mask, parent, self.threshold)
        self.last = (
            q.detach().cpu().numpy(),
            p.cpu().numpy(),
            parent.cpu().numpy(),
            keep.cpu().numpy(),
        )
        return q.masked_fill(~keep, -torch.inf), value, {}

    def record(self, obs, index, chosen, spec, step):
        q, p, parent, keep = (a[index] for a in self.last)
        n, parent = len(obs.legal_actions), int(parent)
        selected = np.sort(q[:n][keep[:n]])
        return {
            "seed": spec[0],
            "seat": spec[2],
            "step": step,
            "family": action_family(obs.legal_actions),
            "chosen": chosen,
            "parent": parent,
            "raw": int(q[:n].argmax()),
            "behavior_probability": float(p[chosen]),
            "parent_probability": float(p[parent]),
            "changed": chosen != parent,
            "candidate_count": n,
            "filtered": int(n - keep[:n].sum()),
            "retained_mean_probability": float(p[:n][keep[:n]].mean()),
            "margin": float(selected[-1] - selected[-2]) if len(selected) > 1 else None,
            "probabilities": p[:n].tolist(),
            "q": q[:n].tolist(),
            "actions": [asdict(a) for a in obs.legal_actions],
        }


def summarize(records):
    changed = [r for r in records if r["changed"]]
    candidates = sum(r["candidate_count"] for r in records)
    return {
        "decisions": len(records),
        "changed_action_rate": len(changed) / max(1, len(records)),
        "filtered_candidate_fraction": sum(r["filtered"] for r in records) / max(1, candidates),
        "selected_behavior_probability": distribution([r["behavior_probability"] for r in records]),
        "retained_mean_probability": distribution(
            [r["retained_mean_probability"] for r in records]
        ),
        "margin": distribution([r["margin"] for r in records if r["margin"] is not None]),
        "singleton_candidates": sum(r["margin"] is None for r in records),
        "overrides": {
            "count": len(changed),
            "new_probability": distribution([r["behavior_probability"] for r in changed]),
            "parent_probability": distribution([r["parent_probability"] for r in changed]),
            "low_support_fraction": {
                str(t): sum(r["behavior_probability"] <= t for r in changed) / max(1, len(changed))
                for t in (0.01, 0.03, 0.05, 0.1)
            },
            "cash_by_probability": {
                f"{lo}:{hi}": {
                    "decisions": len(bucket),
                    "mean_terminal_cash": float(np.mean([r["terminal_cash"] for r in bucket]))
                    if bucket
                    else None,
                }
                for lo, hi in ((0, 0.01), (0.01, 0.03), (0.03, 0.05), (0.05, 0.1), (0.1, 1.01))
                for bucket in [[r for r in changed if lo <= r["behavior_probability"] < hi]]
            },
        },
    }
