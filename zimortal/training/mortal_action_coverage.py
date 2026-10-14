"""Empirical action-template coverage; never counterfactual labels or new training."""

from collections import Counter

import numpy as np

from zimortal.engine import Action
from zimortal.model.encoding import encode_action

from .mortal_diagnostics import distribution


def template_coverage(train, records):
    chosen, eligible = Counter(), Counter()
    for row in train:
        keys = [a.tobytes() for a in row["actions"]]
        family = str(row["family"])
        chosen[family, keys[int(row["chosen"])]] += 1
        eligible.update((family, k) for k in set(keys))
    observations = []
    for r in records:
        if r["chosen"] == r["parent"]:
            continue
        action = Action(**r["actions"][r["chosen"]])
        key = str(r["family"]), encode_action(action, action.player).tobytes()
        observations.append(
            {
                "family": str(r["family"]),
                "probability": r["behavior_probability"],
                "chosen_count": chosen[key],
                "eligible_count": eligible[key],
                "margin": r["margin"] * 100,
            }
        )
    return {
        family: {
            "overrides": len(rows),
            "zero_chosen_template_count": sum(r["chosen_count"] == 0 for r in rows),
            "chosen_count": distribution([r["chosen_count"] for r in rows]),
            "eligible_count": distribution([r["eligible_count"] for r in rows]),
            "margin_for_unseen_template": distribution(
                [r["margin"] for r in rows if r["chosen_count"] == 0]
            ),
            "low_support_nonzero_template_count": sum(
                r["probability"] <= 0.1 and r["chosen_count"] > 0 for r in rows
            ),
        }
        for family in ["all", "discard", "chi/pass", "peng/pass", "hu/pass"]
        for rows in [[r for r in observations if family == "all" or r["family"] == family]]
    }


def clustered_correlation_ci(prediction, target, game_ids, seed=1414):
    """Seed/game cluster resampling; returns None for constant bootstrap draws."""
    prediction, target = np.asarray(prediction), np.asarray(target)
    games = np.asarray(game_ids)
    groups = [np.flatnonzero(games == g) for g in np.unique(games)]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(4096):
        idx = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        if prediction[idx].std() > 1e-10 and target[idx].std() > 1e-10:
            values.append(float(np.corrcoef(prediction[idx], target[idx])[0, 1]))
    return np.quantile(values, [0.025, 0.975]).tolist() if values else None
