"""One fixed-temperature exploration of anchored scores; never raw-Q sampling."""

import numpy as np
import torch

from .mortal_extraction import ExtractionPolicy, family_indices

TEMPERATURE = .5


def sampling_probability(scores, legal, *, conservative_hu=False):
    scores = np.asarray(scores, dtype=np.float64)
    legal = np.asarray(legal, dtype=bool)
    if scores.shape != legal.shape or not legal.any() or not np.isfinite(scores[legal]).all():
        raise ValueError("finite legal anchored scores required")
    result = np.zeros_like(scores)
    if conservative_hu:
        result[int(np.argmax(np.where(legal, scores, -np.inf)))] = 1.
    else:
        logits = scores[legal] / TEMPERATURE
        weights = np.exp(logits - logits.max())
        result[legal] = weights / weights.sum()
    return result


class AnchoredBoltzmann(ExtractionPolicy):
    def __init__(self, q, parent, behavior):
        super().__init__(q, parent, behavior, {"mode": "anchor", "lambda": .3})
        self.context, self.rngs = None, {}

    def prepare(self, specs):
        self.context = [(int(s[0]), int(s[2])) for s in specs]
        if len(set(self.context)) != len(self.context):
            raise ValueError("duplicate sampling context")
        for key in self.context:
            self.rngs.setdefault(key, np.random.default_rng(np.random.SeedSequence([*key, 20261016])))

    def forward(self, features, actions, mask):
        if self.context is None or len(self.context) != len(features):
            raise RuntimeError("per-game sampling context required")
        scores, values, aux = super().forward(features, actions, mask)
        families = family_indices(actions, mask).cpu().tolist()
        legal = mask.cpu().numpy()
        self.actor_probabilities = [sampling_probability(s, m, conservative_hu=f == 3)
                                    for s, m, f in zip(self.last[-1], legal, families, strict=True)]
        self.choices = [int(self.rngs[k].choice(len(p), p=p)) if f != 3 else int(p.argmax())
                        for k, p, f in zip(self.context, self.actor_probabilities, families, strict=True)]
        # Dispatch through the existing greedy simulator. Q/p/reference scores in self.last stay intact.
        dispatch = torch.full_like(scores, -torch.inf)
        dispatch.scatter_(1, torch.tensor(self.choices, device=mask.device)[:, None], 0.)
        return dispatch, values, aux

    def record(self, obs, index, chosen, spec, step):
        if self.context[index] != (spec[0], spec[2]) or chosen != self.choices[index]:
            raise RuntimeError("sampling decision/context mismatch")
        r = super().record(obs, index, chosen, spec, step)
        p = self.actor_probabilities[index][:len(obs.legal_actions)]
        r.update(actor_probabilities=p.tolist(), actual_behavior_probability=float(p[chosen]),
                 sampling_temperature=TEMPERATURE, hu_conservative=r["family"] == "hu/pass")
        return r
