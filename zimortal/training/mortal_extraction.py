"""Fixed-Q action extraction only: no model changes, optimizer, or training."""

from dataclasses import asdict

import numpy as np
import torch
from torch import nn

from zimortal.model.encoding import KINDS

from .mortal_data import action_family
from .mortal_diagnostics import distribution
from .mortal_support import behavior_probability, support_mask

FAMILIES = ("discard", "chi/pass", "peng/pass", "hu/pass")


def family_indices(actions, mask):
    kinds = actions[..., :len(KINDS)].argmax(-1)
    families = torch.zeros(len(mask), dtype=torch.long, device=mask.device)
    for index, name in ((2, "peng"), (1, "chi"), (3, "hu")):
        code = next(i for i, k in enumerate(KINDS) if k.value == name)
        families = torch.where(((kinds == code) & mask).any(-1), index, families)
    return families


def extraction_scores(q, p, mask, parent, families, settings):
    """Q is evaluated on the FULL legal mask before extraction; never mutate it."""
    if not mask.gather(1, parent[:, None]).all():
        raise ValueError("parent must be legal")
    keep = mask.clone()
    scores = q.clone()
    mode = settings["mode"]
    if mode == "baseline":
        keep.zero_().scatter_(1, parent[:, None], True)
    elif mode == "support":
        keep = support_mask(p, mask, parent, .1)
    elif mode == "anchor":
        weight = settings["lambda"]
        if weight < 0 or not np.isfinite(weight):
            raise ValueError("invalid anchor lambda")
        scores = q + weight * torch.log(p + 1e-8)
    elif mode in ("family", "top"):
        if mode == "family":
            for index, name in ((0, "discard"), (1, "chi")):
                if settings[name]:
                    keep = torch.where((families == index)[:, None], mask & (p > .1), keep)
        else:
            ranks = p.masked_fill(~mask, -torch.inf).argsort(dim=-1, descending=True, stable=True)
            positions = torch.empty_like(ranks).scatter_(
                1, ranks, torch.arange(mask.shape[1], device=mask.device).expand_as(ranks)
            )
            for index, name in ((0, "discard_k"), (1, "chi_k")):
                k = settings[name]
                if k not in (2, 3):
                    raise ValueError("top-support k must be 2 or 3")
                keep = torch.where((families == index)[:, None], mask & (positions < k), keep)
        # Peng remains raw. HU/PASS follows frozen parent, not a new hu rule.
        keep = torch.where((families == 3)[:, None], torch.zeros_like(keep), keep)
        keep.scatter_(1, parent[:, None], True)
    elif mode != "raw":
        raise ValueError("unknown extraction mode")
    return scores.masked_fill(~keep, -torch.inf), keep


class ExtractionPolicy(nn.Module):
    feature_version = "huxi"

    def __init__(self, q, parent, behavior, settings):
        super().__init__()
        self.q, self.parent, self.behavior, self.settings = q, parent, behavior, settings
        self.last = None

    def forward(self, features, actions, mask):
        logits, value, _ = self.parent(features, actions, mask)
        p = behavior_probability(logits, actions, mask, self.behavior)
        parent = logits.argmax(-1)
        q = self.q.forward_q(features, actions, mask)[0] if self.q is not None else logits
        scores, keep = extraction_scores(q, p, mask, parent, family_indices(actions, mask), self.settings)
        self.last = tuple(t.detach().cpu().numpy() for t in (q, p, parent, keep, scores))
        return scores, value, {}

    def record(self, obs, index, chosen, spec, step):
        q, p, parent, keep, score = (a[index] for a in self.last)
        n, parent = len(obs.legal_actions), int(parent)
        raw = int(q[:n].argmax())
        ranked = np.sort(q[:n][keep[:n]])
        scored = np.sort(score[:n][keep[:n]])
        alternatives = q[:n][keep[:n] & (np.arange(n) != chosen)]
        return {
            "seed": spec[0], "seat": spec[2], "step": step,
            "family": action_family(obs.legal_actions), "chosen": chosen,
            "parent": parent, "raw": raw, "changed": chosen != parent,
            "behavior_probability": float(p[chosen]), "parent_probability": float(p[parent]),
            "candidate_count": n, "filtered": int(n - keep[:n].sum()),
            "parent_retained": bool(keep[parent]), "raw_retained": bool(keep[raw]),
            "q_selected_minus_parent": float(q[chosen] - q[parent]),
            "q_margin": float(ranked[-1] - ranked[-2]) if len(ranked) > 1 else None,
            "selected_q_margin": float(q[chosen] - alternatives.max()) if len(alternatives) else None,
            "score_margin": float(scored[-1] - scored[-2]) if len(scored) > 1 else None,
            "q": q[:n].tolist(), "scores": score[:n][keep[:n]].tolist(),
            "probabilities": p[:n].tolist(),
            "actions": [asdict(a) for a in obs.legal_actions],
        }


def summarize_families(records):
    def summary(rows):
        changed = [r for r in rows if r["changed"]]
        return {
            "decisions": len(rows), "overrides": len(changed),
            "changed_action_rate": len(changed) / max(1, len(rows)),
            "parent_selection_rate": sum(not r["changed"] for r in rows) / max(1, len(rows)),
            "parent_candidate_retention_rate": sum(r["parent_retained"] for r in rows) / max(1, len(rows)),
            "raw_candidate_retention_rate": sum(r["raw_retained"] for r in rows) / max(1, len(rows)),
            "low_support_among_overrides": sum(r["behavior_probability"] <= .1 for r in changed) / len(changed) if changed else None,
            "low_support_overrides_per_decision": sum(r["behavior_probability"] <= .1 for r in changed) / max(1, len(rows)),
            "q_margin": distribution([r["q_margin"] for r in rows if r["q_margin"] is not None]),
            "selected_q_margin": distribution([r["selected_q_margin"] for r in rows if r["selected_q_margin"] is not None]),
            "score_margin": distribution([r["score_margin"] for r in rows if r["score_margin"] is not None]),
            "q_selected_minus_parent": distribution([r["q_selected_minus_parent"] for r in rows]),
        }
    return {f: summary([r for r in records if f == "all" or r["family"] == f]) for f in ("all", *FAMILIES)}
