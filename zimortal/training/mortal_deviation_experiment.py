"""One 3000-seed frozen-actor validation. Zero updates, no parameter selection."""

import argparse
import gzip
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from zimortal.engine import GameState

from .active import atomic_json
from .mortal_credit import audit_screening
from .mortal_deviation import DeviationPolicy
from .mortal_deviation_stats import MARGIN_EDGES, P_EDGES, Q_EDGES, deviation_statistics
from .mortal_extraction_experiment import CHECKPOINT
from .mortal_offline import BASELINE
from .mortal_sarsa import parameter_hash
from .online_rl import digest, paired_summary, play, source_hashes
from .runtime import load_model

BUDGETS = {"baseline": 0, "anchor-all": None, "anchor-first": 1, "anchor-max2": 2}


def audit_batch(batch, pending, budget):
    state, key = pending["state"], pending["key"]
    count = 0
    previous = -1
    for r in batch:
        if (r["seed"], r["seat"]) != key or r["step"] <= previous:
            raise RuntimeError("decision game/step mismatch")
        previous = r["step"]
        actual = json.loads(json.dumps(asdict(state.history[r["step"] - 1])))
        recorded = json.loads(json.dumps(r["actions"][r["chosen"]]))
        if recorded != actual or r["changed"] != (r["chosen"] != r["parent"]):
            raise RuntimeError("recorded deviation differs from actual history")
        if r["deviation_count_before"] != count:
            raise RuntimeError("deviation counter prefix mismatch")
        available = budget is None or count < budget
        expected = r["unlimited_anchor_choice"] if available else r["parent"]
        if r["chosen"] != expected:
            raise RuntimeError("actual history violates budget")
        q, p = np.array(r["q"]), np.array(r["probabilities"])
        scores = q + .3 * np.log(p + 1e-8)
        if scores[r["unlimited_anchor_choice"]] < scores.max() - 1e-5:
            raise RuntimeError("anchor score/proposal mismatch")
        if not np.isclose(r["q_selected_minus_parent"], q[r["chosen"]] - q[r["parent"]], atol=1e-5):
            raise RuntimeError("raw Q difference mismatch")
        count += int(r["changed"])
        if r["deviation_count_after"] != count or (budget is not None and count > budget):
            raise RuntimeError("actual deviation budget exceeded")
        cash = state.settlement.payments[key[1]] if state.settlement else 0
        if cash != r["terminal_cash"]:
            raise RuntimeError("deviation terminal cash mismatch")
    return count


def run(directory):
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    prior = json.loads(Path("data/generated/mortal-extraction-v1/config.json").read_text())
    for p, sha in prior["frozen_sha256"].items():
        required = p.startswith(("zimortal/engine/", "zimortal/model/")) or p in (CHECKPOINT, BASELINE)
        if required and digest(p) != sha:
            raise RuntimeError("frozen rules/model/checkpoint changed")
    frozen = source_hashes() | {CHECKPOINT: digest(CHECKPOINT), BASELINE: digest(BASELINE)}
    config = {
        "start": 1150000000, "seeds": 3000, "budgets": BUDGETS,
        "checkpoint": CHECKPOINT, "behavior": prior["behavior"], "lambda": .3, "eps": 1e-8,
        "frozen_sha256": frozen,
        "p_edges": list(P_EDGES), "margin_edges_Q_units": [str(x) for x in MARGIN_EDGES],
        "q_delta_edges_Q_units": [str(x) for x in Q_EDGES],
        "bootstrap": "10000 seed-cluster replicates; all 3 seats paired for primary",
        "simultaneous_diagnostic_alpha": .05 / 3,
        "primary": "anchor-all", "secondary_predeclared": ["anchor-first", "anchor-max2"],
        "no_test_selection": True, "training_updates": 0, "online_started": False,
        "bins_descriptive_not_action_EV": True,
        "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0),
    }
    atomic_json(root / "config.json", config)
    specs = [(s, s % 3, seat, 0) for s in range(1150000000, 1150003000) for seat in range(3)]
    atomic_json(root / "seed-lock.json", specs)
    q, parent = [load_model(p, "cuda").requires_grad_(False).eval() for p in (CHECKPOINT, BASELINE)]
    state_hashes = {"q": parameter_hash(q), "parent": parameter_hash(parent)}
    outcomes, statistics, all_counts = {}, {}, {}
    baseline = None
    for name, budget in BUDGETS.items():
        policy = DeviationPolicy(q, parent, prior["behavior"], budget).eval()
        records, counts, pending = [], {}, {}
        atomic_json(root / "progress.json", {"policy": name, "games": len(specs), "training_updates": 0})
        with gzip.open(root / f"screen-{name}-replays.jsonl.gz", "wt") as replays, gzip.open(root / f"decisions-{name}.jsonl.gz", "wt") as decisions:
            def sink(e, pending=pending):
                replays.write(json.dumps(e) + "\n")
                pending.update(state=GameState.deserialize(e["state"]), key=(e["seed"], e["seat"]))
            def decision_sink(batch, pending=pending, budget=budget, policy=policy, counts=counts, records=records):
                key = pending["key"]
                count = audit_batch(batch, pending, budget)
                if count != policy.counts.get(key, 0):
                    raise RuntimeError("policy and audited budget mismatch")
                counts[key] = count
                for r in batch:
                    decisions.write(json.dumps(r) + "\n")
                    records.append({k: v for k, v in r.items() if k not in ("q", "scores", "probabilities", "actions")})
                pending.clear()
            rows, games, errors = play(
                policy, [parent], specs, device="cuda", slots=32, training=False,
                replay_sink=sink, decision_recorder=policy.record, decision_sink=decision_sink,
                decision_context=policy.prepare,
            )
        if rows or errors or len(games) != len(specs) or len(counts) != len(specs):
            atomic_json(root / "failures.json", errors)
            raise RuntimeError("evaluation incomplete or unexpectedly produced training samples")
        indexed = {(g["seed"], g["seat"]): g for g in games}
        if baseline is None:
            baseline = indexed
        for k, g in indexed.items():
            if counts[k] == 0 and g["history_sha256"] != baseline[k]["history_sha256"]:
                raise RuntimeError("zero-deviation history differs from paired baseline")
        outcomes[name] = games
        all_counts[name] = counts
        statistics[name] = deviation_statistics(records, games, baseline, counts)
        atomic_json(root / "screening-outcomes.json", outcomes)
        atomic_json(root / "statistics.json", statistics)
        atomic_json(root / f"game-deviations-{name}.json", [
            {"seed": k[0], "seat": k[1], "deviations": n, "cash": indexed[k]["cash"],
             "baseline_cash": baseline[k]["cash"]} for k, n in sorted(counts.items())
        ])
        if parameter_hash(q) != state_hashes["q"] or parameter_hash(parent) != state_hashes["parent"]:
            raise RuntimeError("network weights or buffers changed")
        for p, sha in frozen.items():
            if digest(p) != sha:
                raise RuntimeError("frozen source/checkpoint changed")
    unrestricted = {(g["seed"], g["seat"]): g for g in outcomes["anchor-all"]}
    for name, budget in (("anchor-first", 1), ("anchor-max2", 2)):
        for g in outcomes[name]:
            key = (g["seed"], g["seat"])
            if all_counts["anchor-all"][key] <= budget and g["history_sha256"] != unrestricted[key]["history_sha256"]:
                raise RuntimeError("nonbinding budget changed history")
    audit = audit_screening(root)
    comparisons = {n: paired_summary(g, outcomes["baseline"], 1515, alpha=.05 / 3) for n, g in outcomes.items()}
    atomic_json(root / "result.json", {
        "status": "complete_deviation_validation_stop", "comparisons": comparisons,
        "budget_contrasts": {n: paired_summary(outcomes[n], outcomes["anchor-all"], 1515) for n in ("anchor-first", "anchor-max2")},
        "statistics": statistics, "audit": audit, "network_state_sha256": state_hashes,
        "zero_deviation_history_verified": True, "nonbinding_budget_history_verified": True,
        "training_updates": 0, "online_started": False, "baseline_replaced": False,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="data/generated/mortal-deviation-v1")
    run(parser.parse_args().directory)
