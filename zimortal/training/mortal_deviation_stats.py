"""Predeclared descriptive cohorts; terminal returns are not causal action values."""

from collections import defaultdict
from itertools import pairwise

import numpy as np

from .mortal_diagnostics import distribution

P_EDGES = (0., .1, .2, .3, .5, 1.0000001)
MARGIN_EDGES = (0., .001, .003, .01, .03, .1, float("inf"))
Q_EDGES = (-float("inf"), 0., .01, .03, .1, .3, float("inf"))


def bin_index(value, edges):
    if not np.isfinite(value):
        raise ValueError("nonfinite audit value")
    index = int(np.searchsorted(edges, value, side="right") - 1)
    if not 0 <= index < len(edges) - 1:
        raise ValueError("value outside predeclared bins")
    return index


def cluster_cohort(games, baseline, *, bootstrap=10000):
    if not games:
        return {"games": 0, "seeds": 0, "mean_cash": None, "paired_delta": None, "seed_bootstrap_95": None}
    clusters = defaultdict(lambda: [0., 0])
    for g in games:
        clusters[g["seed"]][0] += g["cash"] - baseline[(g["seed"], g["seat"])]["cash"]
        clusters[g["seed"]][1] += 1
    values = np.array([clusters[s] for s in sorted(clusters)])
    ci = None
    if len(values) > 1 and (values[:, 0] == 0).all():
        ci = [0., 0.]
    elif len(values) > 1:
        rng = np.random.default_rng(1515)
        boot = []
        for start in range(0, bootstrap, 512):
            drawn = values[rng.integers(len(values), size=(min(512, bootstrap - start), len(values)))].sum(1)
            boot.extend((drawn[:, 0] / drawn[:, 1]).tolist())
        ci = np.quantile(boot, [.025, .975]).tolist()
    return {
        "games": len(games), "seeds": len(values),
        "mean_cash": float(np.mean([g["cash"] for g in games])),
        "mean_baseline_cash_same_games": float(np.mean([baseline[(g["seed"], g["seat"])]["cash"] for g in games])),
        "paired_delta": float(values[:, 0].sum() / values[:, 1].sum()), "seed_bootstrap_95": ci,
    }


def deviation_statistics(records, games, baseline, counts):
    indexed = {(g["seed"], g["seat"]): g for g in games}
    changed = [r for r in records if r["changed"]]
    def group(rows):
        keys = {(r["seed"], r["seat"]) for r in rows}
        return {
            "decisions": len(rows),
            "unique_game_cash": cluster_cohort([indexed[k] for k in sorted(keys)], baseline),
            "p_behavior": distribution([r["behavior_probability"] for r in rows]),
            "q_delta": distribution([r["q_selected_minus_parent"] for r in rows]),
            "anchor_margin": distribution([r["unlimited_anchor_margin"] for r in rows]),
            "decision_weighted_terminal_cash": distribution([r["terminal_cash"] for r in rows]),
            "not_counterfactual_action_EV": True,
        }
    family = {f: group([r for r in changed if r["family"] == f]) for f in ("discard", "chi/pass", "peng/pass", "hu/pass")}
    buckets = {}
    for name, field, edges in (("p_behavior", "behavior_probability", P_EDGES),
                               ("anchor_margin", "unlimited_anchor_margin", MARGIN_EDGES),
                               ("q_delta", "q_selected_minus_parent", Q_EDGES)):
        buckets[name] = {f"[{lo},{hi})": group([r for r in changed if bin_index(r[field], edges) == i])
                         for i, (lo, hi) in enumerate(pairwise(edges))}
    count_cohorts = {}
    for label, predicate in (("zero", lambda n: n == 0), ("once", lambda n: n == 1),
                             ("multiple", lambda n: n > 1)):
        count_cohorts[label] = cluster_cohort([g for k, g in indexed.items() if predicate(counts[k])], baseline)
    return {
        "deviations": len(changed), "decisions": len(records),
        "deviation_rate": len(changed) / max(1, len(records)),
        "blocked_proposals": sum(r["blocked_by_budget"] for r in records),
        "family": family, "buckets": buckets, "count_cohorts": count_cohorts,
        "count_histogram": {str(n): sum(v == n for v in counts.values()) for n in sorted(set(counts.values()))},
        "descriptive_conditioning_not_causal": True,
    }
