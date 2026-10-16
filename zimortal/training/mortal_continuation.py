"""One real training root per game; only the subsequent continuation differs."""

import numpy as np
import torch

from .mortal_boltzmann import TEMPERATURE, AnchoredBoltzmann, sampling_probability
from .mortal_extraction import ExtractionPolicy, family_indices


class ContinuationPolicy(AnchoredBoltzmann):
    def __init__(self, q, parent, behavior, *, greedy_continuation):
        super().__init__(q, parent, behavior)
        self.greedy_continuation = greedy_continuation
        self.root_steps = {}

    def forward(self, features, actions, mask):
        if self.context is None or len(self.context) != len(features):
            raise RuntimeError("per-game context required")
        scores, value, aux = ExtractionPolicy.forward(self, features, actions, mask)
        families = family_indices(actions, mask).cpu().tolist()
        self.actor_probabilities, self.choices, self.root_flags, self.modes = [], [], [], []
        for key, s, legal, family in zip(self.context, self.last[-1], mask.cpu().numpy(), families, strict=True):
            done = key in self.root_steps
            root = not done and family != 3
            greedy = family == 3 or (self.greedy_continuation and done)
            p = sampling_probability(s, legal, conservative_hu=greedy)
            choice = int(p.argmax()) if greedy else int(self.rngs[key].choice(len(p), p=p))
            self.actor_probabilities.append(p)
            self.choices.append(choice)
            self.root_flags.append(root)
            self.modes.append('hu_greedy' if family == 3 else 'greedy_continuation' if greedy else 'boltzmann')
        dispatch = torch.full_like(scores, -torch.inf)
        dispatch.scatter_(1, torch.tensor(self.choices, device=mask.device)[:, None], 0.)
        return dispatch, value, aux

    def record(self, obs, index, chosen, spec, step):
        key = (spec[0], spec[2])
        if self.context[index] != key or chosen != self.choices[index]:
            raise RuntimeError("continuation decision/context mismatch")
        root = self.root_flags[index]
        if root and key in self.root_steps:
            raise RuntimeError("duplicate training root")
        r = ExtractionPolicy.record(self, obs, index, chosen, spec, step)
        p = self.actor_probabilities[index][:len(obs.legal_actions)]
        greedy = int(self.last[-1][index][:len(p)].argmax())
        r.update(actor_probabilities=p.tolist(), actual_behavior_probability=float(p[chosen]),
                 sampling_temperature=TEMPERATURE if self.modes[index] == 'boltzmann' else None,
                 execution_mode=self.modes[index], is_training_root=root,
                 switch_trigger=bool(root and self.greedy_continuation),
                 changed_from_anchored_greedy=chosen != greedy,
                 anchored_greedy=greedy, hu_conservative=r['family'] == 'hu/pass')
        if root:
            # Recorder immediately precedes step. A failed step rejects the whole game;
            # collect/audit independently verify this action in the completed history.
            self.root_steps[key] = step
        return r


def first_roots(rows):
    roots = {}
    for row in rows:
        if row['family'] == 'hu/pass':
            continue
        key = (int(row['seed']), int(row['seat']))
        if key not in roots or row['history_step'] < roots[key]['history_step']:
            roots[key] = row
    return roots


def matched_roots(a, b):
    """Match only root availability, never chosen action, return, CI or outcome."""
    roots = {'A': first_roots(a), 'B': first_roots(b)}
    common = sorted(roots['A'].keys() & roots['B'].keys())
    if not common:
        raise ValueError('no common training roots')
    # Copy the small selection so it does not pin every full-game NPZ backing array.
    selected = {arm: [{k: v.copy() if isinstance(v, np.ndarray) else v for k, v in roots[arm][key].items()}
                      for key in common] for arm in roots}
    manifest = {'common_seed_seat': common, 'selection': 'first non-HU multi-choice decision; no return filtering',
                'unmatched_A_only': sorted(roots['A'].keys() - roots['B'].keys()),
                'unmatched_B_only': sorted(roots['B'].keys() - roots['A'].keys()),
                'history_steps': {arm: [int(r['history_step']) for r in selected[arm]] for arm in roots}}
    return selected, manifest
