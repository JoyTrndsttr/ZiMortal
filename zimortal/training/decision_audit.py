"""Descriptive visible-decision audits, never action-EV training labels."""

from collections import Counter, defaultdict
from dataclasses import asdict

import numpy as np

from zimortal.engine import ActionType as A
from zimortal.engine.scoring import meld_huxi


def category(observation):
    kinds = {a.kind for a in observation.legal_actions}
    if A.HU in kinds:
        return "hu_pass"
    if A.PENG in kinds:
        return "peng_pass"
    if A.CHI in kinds:
        return "chi_bi_pass"
    return None


def record(observation, chosen, parent, probabilities, *, spec, step, immediate_hu_cash=None):
    """All policy diagnostics are computed from this player's Observation."""
    actions = observation.legal_actions
    return {
        "category": category(observation),
        "seed": spec[0],
        "dealer": spec[1],
        "learner_seat": spec[2],
        "player": observation.player,
        "opponent": spec[3],
        "step": step,
        "phase": observation.phase,
        "remaining_tiles": observation.remaining_tiles,
        "hand": list(observation.hand),
        "exposed_huxi": sum(meld_huxi(m) for m in observation.players[observation.player].melds),
        "chosen": asdict(actions[chosen]),
        "parent": asdict(actions[parent]),
        "different_from_parent": chosen != parent,
        "options": [
            {"action": asdict(a), "probability": float(p)}
            for a, p in zip(actions, probabilities, strict=True)
        ],
        "immediate_hu_cash": immediate_hu_cash,
    }


def summarize(records, *, stochastic, examples=6):
    groups = defaultdict(list)
    for row in records:
        groups[row["category"]].append(row)
    result = {
        "sampling": "on-policy stochastic" if stochastic else "greedy evaluation",
        "interpretation": "descriptive outcomes; action groups are not causal Q or EV labels",
        "categories": {},
    }
    for name in ("chi_bi_pass", "peng_pass", "hu_pass"):
        rows = groups[name]
        by_action = {}
        for kind in sorted({r["chosen"]["kind"] for r in rows}):
            selected = [r for r in rows if r["chosen"]["kind"] == kind]
            by_action[kind] = {
                "decisions": len(selected),
                "mean_terminal_cash": float(np.mean([r["terminal_cash"] for r in selected])),
                "win_rate": float(np.mean([r["winner"] == r["player"] for r in selected])),
            }
        passed_hu = [r for r in rows if name == "hu_pass" and r["chosen"]["kind"] != "hu"]
        differences = [r for r in rows if r["different_from_parent"]]
        # First collect concrete changes, then distinct seeds rather than many
        # correlated decisions from a single game. Retain full legal chi/bi options.
        ranked = sorted(
            rows,
            key=lambda r: (
                not r["different_from_parent"],
                r["terminal_cash"] - (r["immediate_hu_cash"] or 0),
                r["seed"],
                r["step"],
            ),
        )
        cases, seen = [], set()
        for row in ranked:
            key = (row["seed"], row["player"], row["opponent"])
            if key not in seen and len(cases) < examples:
                seen.add(key)
                cases.append(row)
        result["categories"][name] = {
            "decisions": len(rows),
            "distinct_games": len({(r["seed"], r["opponent"]) for r in rows}),
            "different_from_parent": len(differences),
            "parent_to_chosen": dict(
                Counter(f"{r['parent']['kind']}->{r['chosen']['kind']}" for r in differences)
            ),
            "actions": by_action,
            "passed_hu": len(passed_hu),
            "pass_hu_worse_than_immediate": sum(
                r["terminal_cash"] < r["immediate_hu_cash"] for r in passed_hu
            ),
            "mean_realized_cash_minus_immediate_hu": (
                float(np.mean([r["terminal_cash"] - r["immediate_hu_cash"] for r in passed_hu]))
                if passed_hu
                else None
            ),
            "examples": cases,
        }
    return result
