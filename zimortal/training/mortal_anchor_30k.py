"""Predeclared 30k paired anchor-all evaluation; no training or selection."""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import torch

from zimortal.engine import GameState

from .active import atomic_json
from .mortal_credit import audit_screening
from .mortal_deviation import DeviationPolicy
from .mortal_deviation_experiment import audit_batch
from .mortal_deviation_stats import cluster_cohort
from .mortal_extraction_experiment import CHECKPOINT
from .mortal_offline import BASELINE
from .mortal_sarsa import parameter_hash
from .online_rl import digest, play, source_hashes
from .runtime import load_model

BUDGETS = {"baseline": 0, "anchor-all": None}
START = 1160000000
SEEDS = 30000


def summarize(records, games, baseline, counts, decisions):
    indexed = {(g["seed"], g["seat"]): g for g in games}
    if len(indexed) != len(games) or indexed.keys() != baseline.keys() or indexed.keys() != counts.keys():
        raise RuntimeError("incomplete or duplicate paired games")
    seats = {}
    for seed, seat in indexed:
        seats.setdefault(seed, set()).add(seat)
    if any(s != {0, 1, 2} for s in seats.values()):
        raise RuntimeError("primary comparison requires all three seats per seed")
    if sum(counts.values()) != len(records):
        raise RuntimeError("deviation records/count mismatch")
    families = {}
    for family in ("discard", "chi/pass", "peng/pass", "hu/pass"):
        rows = [r for r in records if r["family"] == family]
        keys = {(r["seed"], r["seat"]) for r in rows}
        families[family] = {
            "deviations": len(rows),
            "unique_game_cash": cluster_cohort([indexed[k] for k in sorted(keys)], baseline),
            "descriptive_not_causal": True,
        }
    return {
        "primary": cluster_cohort(games, baseline),
        "decisions": decisions, "deviations": len(records),
        "deviation_rate": len(records) / max(1, decisions),
        "deviating_games": sum(n > 0 for n in counts.values()),
        "deviating_game_fraction": sum(n > 0 for n in counts.values()) / len(games),
        "count_histogram": dict(sorted(Counter(counts.values()).items())),
        "family": families,
        "previous_rate": 0.015674444,
    }


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
        "start": START, "seeds": SEEDS, "budgets": BUDGETS,
        "checkpoint": CHECKPOINT, "behavior": prior["behavior"], "lambda": .3, "eps": 1e-8,
        "frozen_sha256": frozen,
        "bootstrap": "10000 bounded-memory seed-cluster replicates; all 3 seats paired",
        "primary": "anchor-all minus baseline", "confidence": .95,
        "no_test_selection": True, "training_updates": 0, "online_started": False,
        "bins_descriptive_not_action_EV": True,
        "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0),
    }
    atomic_json(root / "config.json", config)
    specs = [(s, s % 3, seat, 0) for s in range(START, START + SEEDS) for seat in range(3)]
    atomic_json(root / "seed-lock.json", specs)
    q, parent = [load_model(p, "cuda").requires_grad_(False).eval() for p in (CHECKPOINT, BASELINE)]
    state_hashes = {"q": parameter_hash(q), "parent": parameter_hash(parent)}
    outcomes, statistics = {}, {}
    baseline = None
    for name, budget in BUDGETS.items():
        policy = DeviationPolicy(q, parent, prior["behavior"], budget).eval()
        records, counts, pending = [], {}, {}
        totals = {"decisions": 0}
        atomic_json(root / "progress.json", {"policy": name, "games": len(specs), "training_updates": 0})
        with gzip.open(root / f"screen-{name}-replays.jsonl.gz", "wt") as replays, gzip.open(root / f"decisions-{name}.jsonl.gz", "wt") as decisions:
            def sink(e, pending=pending):
                replays.write(json.dumps(e) + "\n")
                pending.update(state=GameState.deserialize(e["state"]), key=(e["seed"], e["seat"]))
            def decision_sink(batch, pending=pending, budget=budget, policy=policy, counts=counts, records=records, totals=totals, name=name):
                key = pending["key"]
                count = audit_batch(batch, pending, budget)
                if count != policy.counts.get(key, 0):
                    raise RuntimeError("policy and audited budget mismatch")
                counts[key] = count
                for r in batch:
                    decisions.write(json.dumps(r) + "\n")
                    totals["decisions"] += 1
                    if r["changed"]:
                        records.append({k: v for k, v in r.items() if k not in ("q", "scores", "probabilities", "actions")})
                pending.clear()
                if len(counts) % 250 == 0:
                    atomic_json(root / "progress.json", {
                        "policy": name, "completed": len(counts), "games": len(specs),
                        "decisions": totals["decisions"], "deviations": len(records),
                        "training_updates": 0,
                    })
                    if len(counts) % 3000 == 0:
                        print(name, len(counts), "/", len(specs), flush=True)
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
        statistics[name] = summarize(records, games, baseline, counts, totals["decisions"])
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
    atomic_json(root / "progress.json", {"phase": "independent_full_replay_audit", "training_updates": 0})
    audit = audit_screening(root)
    for path, sha in frozen.items():
        if digest(path) != sha:
            raise RuntimeError("frozen source/checkpoint changed during audit")
    atomic_json(root / "result.json", {
        "status": "complete_30k_validation_stop", "statistics": statistics,
        "audit": audit, "network_state_sha256": state_hashes,
        "zero_deviation_history_verified": True,
        "training_updates": 0, "online_started": False, "baseline_replaced": False,
    })
    atomic_json(root / "progress.json", {"phase": "complete_stop", "games": audit["games"], "training_updates": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="data/generated/mortal-anchor-30k-v1")
    run(parser.parse_args().directory)
