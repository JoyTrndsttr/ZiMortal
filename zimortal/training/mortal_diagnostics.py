"""Descriptive offline diagnostics; unchosen actions have no counterfactual labels."""

import numpy as np


def distribution(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {"count": 0}
    return {
        "count": len(a),
        "mean": float(a.mean()),
        "std": float(a.std()),
        **{f"p{p}": float(np.percentile(a, p)) for p in (5, 25, 50, 75, 95)},
        "zero_fraction": float(np.mean(a == 0)),
        "large_positive_fraction": float(np.mean(a >= 1)),
        "large_negative_fraction": float(np.mean(a <= -1)),
    }


def metrics(p, t):
    p, t = np.asarray(p), np.asarray(t)
    return {
        "count": len(p),
        "mae": float(np.mean(abs(p - t))),
        "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
        "correlation": float(np.corrcoef(p, t)[0, 1])
        if p.std() > 1e-10 and t.std() > 1e-10
        else None,
    }


def diagnostics(rows, predictions, targets, q_rows, thresholds=(0, 0.02, 0.05, 0.1, 0.2)):
    margins, changes, gains = [], [], []
    for row, q in zip(rows, q_rows, strict=True):
        ordered = np.sort(q)
        margins.append(float(ordered[-1] - ordered[-2]))
        changes.append(int(q.argmax()) != int(row["parent_greedy"]))
        gains.append(float(q.max() - q[int(row["parent_greedy"])]))
    margins, changes = np.asarray(margins), np.asarray(changes, dtype=bool)
    calibration = []
    for indices in np.array_split(np.argsort(predictions, kind="stable"), 5):
        if len(indices):
            calibration.append(
                {
                    "count": len(indices),
                    "predicted_mean": float(predictions[indices].mean()),
                    "target_mean": float(targets[indices].mean()),
                }
            )
    family = {}
    for name in sorted({str(r["family"]) for r in rows}):
        mask = np.array([str(r["family"]) == name for r in rows])
        family[name] = metrics(predictions[mask], targets[mask])
    return {
        "chosen_q_std": float(predictions.std()),
        "calibration": calibration,
        "per_family": family,
        "returns": distribution(targets),
        "q_margin": {
            "all": distribution(margins),
            "changed": distribution(margins[changes]),
            "unchanged": distribution(margins[~changes]),
        },
        "parent_q_gain": distribution(gains),
        "gate_rates": {str(t): float(np.mean(np.asarray(gains) > t)) for t in thresholds},
    }


def select_gate(development, budget=0.1):
    # Predeclared change budget, never selected by held-out rewards.
    return next((float(t) for t, rate in development["gate_rates"].items() if rate <= budget), 0.2)


def coverage(rows, game_cash=None):
    families = {}
    for name in sorted({str(r["family"]) for r in rows}):
        group = [r for r in rows if str(r["family"]) == name]
        changed = sum(int(r["chosen"]) != int(r["parent_greedy"]) for r in group)
        families[name] = {
            "decisions": len(group),
            "changed": changed,
            "changed_rate": changed / len(group),
            "behavior_probability_mean": float(
                np.mean([r["behavior_probabilities"][int(r["chosen"])] for r in group])
            ),
            "entropy_mean": float(
                np.mean(
                    [
                        -np.sum(p[p > 0] * np.log(p[p > 0]))
                        for p in [r["behavior_probabilities"] for r in group]
                    ]
                )
            ),
            "parent_top1_probability_mean": float(
                np.mean([r["behavior_probabilities"][int(r["parent_greedy"])] for r in group])
            ),
            "second_rank_rate": float(np.mean([int(r["behavior_rank"]) == 2 for r in group])),
            "long_tail_rate": float(np.mean([int(r["behavior_rank"]) >= 3 for r in group])),
        }
    passed = [r for r in rows if r["family"] == "hu/pass" and r["chosen_kind"] == "pass"]
    return {
        "families": families,
        "changed_rate": sum(v["changed"] for v in families.values()) / len(rows),
        "hu_pass": {
            "decisions": len(passed),
            "terminal_cash": distribution([r["terminal_cash"] for r in passed]),
            "note": "observational realized cash, not causal HU versus PASS advantage",
        },
        "game_returns": distribution(
            list(game_cash.values())
            if game_cash is not None
            else list({int(r["game_id"]): float(r["terminal_cash"]) for r in rows}.values())
        ),
    }
