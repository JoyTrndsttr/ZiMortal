"""Predeclared 50k single-line online verification; final candidate is Gen10 only."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from .active import atomic_json
from .mortal_boltzmann import AnchoredBoltzmann
from .mortal_boltzmann_experiment import behavior_summary
from .mortal_credit import audit_screening
from .mortal_data import audit_dataset, load_rows, save_shard
from .mortal_deviation_stats import cluster_cohort
from .mortal_diagnostics import distribution
from .mortal_extraction import ExtractionPolicy
from .mortal_extraction_experiment import CHECKPOINT
from .mortal_offline import BASELINE, regression, tensors
from .mortal_online import checkpoint, collect, make_optimizer, update
from .mortal_online_experiment import assert_frozen
from .mortal_sarsa import parameter_hash
from .online_rl import digest, source_hashes
from .runtime import load_model

GENERATIONS = 10
GAMES = 5000


def seed_specs(generation=None, *, evaluation=False, holdout=False):
    if holdout:
        if generation is not None or evaluation:
            raise ValueError("holdout has its own fixed seed block")
        start, count = 1230000000, 5000
    else:
        if generation not in range(1, GENERATIONS + 1):
            raise ValueError("generation must be 1..10")
        start = (1220000000 if evaluation else 1210000000) + generation * 100000
        count = 1000 if evaluation else GAMES
    if evaluation or holdout:
        return [(s, s % 3, seat, 0) for s in range(start, start + count) for seat in range(3)]
    return [(start + i, i % 3, (i // 3) % 3, 0) for i in range(count)]


def safety_stop(results):
    return len(results) >= 2 and all(
        r["cash"]["anchor"]["seed_bootstrap_95"][1] < -5 for r in results[-2:]
    )


def parameter_change(before, model):
    current = dict(model.named_parameters())
    norm = sum(float(p.double().square().sum()) for p in before.values()) ** .5
    delta = sum(float((current[k].detach().cpu().double() - p.double()).square().sum())
                for k, p in before.items()) ** .5
    return {"previous_parameter_l2": norm, "delta_l2": delta, "relative_delta_l2": delta / max(norm, 1e-12)}


def q_probe(model, rows, device="cuda"):
    """Same current-generation inputs before/after update; no labels/backprop/replay."""
    result = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), 256):
            batch = rows[start:start + 256]
            x, a, mask, _, _ = tensors(batch, device, 1.)
            q = model.forward_q(x, a, mask)[0]
            result.extend(v[:len(r["actions"])].cpu().numpy().copy() for v, r in zip(q, batch, strict=True))
    return result


def q_drift(before, after, rows):
    if not (len(before) == len(after) == len(rows)):
        raise ValueError("Q drift inputs differ")
    delta = np.concatenate([b - a for a, b in zip(before, after, strict=True)])
    chosen = np.array([b[r["chosen"]] - a[r["chosen"]] for a, b, r in zip(before, after, rows, strict=True)])
    return {"inputs": len(rows), "legal_q_delta": distribution(delta),
            "legal_q_rmse": float(np.sqrt(np.mean(delta ** 2))),
            "chosen_q_delta": distribution(chosen),
            "raw_argmax_change_rate": float(np.mean([a.argmax() != b.argmax() for a, b in zip(before, after, strict=True)])),
            "scope": "current generation training inputs only; descriptive drift, not generalization"}


def evidence_summary(directory):
    """Read existing action evidence, never regenerate trajectories or infer hidden state."""
    values = {}
    for path in sorted(Path(directory).rglob("decisions-*.jsonl.gz")):
        with gzip.open(path, "rt") as stream:
            for line in stream:
                r = json.loads(line)
                q = np.asarray(r["q"], float)
                if not len(q):  # baseline has no Q; not a Q diagnostic
                    continue
                p = np.asarray(r.get("actor_probabilities", []), float)
                entropy = float(-np.sum(p[p > 0] * np.log(p[p > 0]))) if len(p) else 0.
                for family in ("all", r["family"]):
                    v = values.setdefault(family, {k: [] for k in ("entropy", "chosen_centered_q", "raw_centered_q")})
                    v["entropy"].append(entropy)
                    v["chosen_centered_q"].append(float(q[r["chosen"]] - q.mean()))
                    v["raw_centered_q"].append(float(q.max() - q.mean()))
    return {f: {k: distribution(v) for k, v in metrics.items()} for f, metrics in values.items()}


def evaluate(root, specs, model, initial, parent, reference, frozen, parent_hash):
    root.mkdir()
    states = [parameter_hash(m) for m in (model, initial)]
    policies = {"baseline": ExtractionPolicy(None, parent, reference, {"mode": "baseline", "lambda": .3}),
                "initial": ExtractionPolicy(initial, parent, reference, {"mode": "anchor", "lambda": .3}),
                "anchor": ExtractionPolicy(model, parent, reference, {"mode": "anchor", "lambda": .3}),
                "raw": ExtractionPolicy(model, parent, reference, {"mode": "raw", "lambda": .3})}
    outcomes, behavior = {}, {}
    for name, policy in policies.items():
        atomic_json(root / "phase.json", {"policy": name, "games": len(specs)})
        games, records, _ = collect(policy.eval(), parent, specs, root, name, device="cuda")
        outcomes[name] = games
        behavior[name] = behavior_summary(records)
        atomic_json(root / "screening-outcomes.json", outcomes)
        del records
    audit = audit_screening(root)
    baseline = {(g["seed"], g["seat"]): g for g in outcomes["baseline"]}
    original = {(g["seed"], g["seat"]): g for g in outcomes["initial"]}
    cash = {n: cluster_cohort(g, baseline) for n, g in outcomes.items()}
    contrasts = {"anchor_minus_initial": cluster_cohort(outcomes["anchor"], original),
                 "raw_minus_anchor": cluster_cohort(outcomes["raw"], {(g["seed"], g["seat"]): g for g in outcomes["anchor"]})}
    assert_frozen(frozen, parent, parent_hash)
    if states != [parameter_hash(m) for m in (model, initial)]:
        raise RuntimeError("evaluation modified model/BN")
    result = {"cash": cash, "contrasts": contrasts, "behavior": behavior, "audit": audit}
    atomic_json(root / "result.json", result)
    return result


def run(directory):
    torch.set_num_threads(2)
    torch.manual_seed(20261015)  # retain previous online initialization/RNG recipe
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    old = json.loads(Path("data/generated/mortal-online-v1/config.json").read_text())
    for path, sha in old["frozen_sha256"].items():
        if (path.startswith(("zimortal/engine/", "zimortal/model/")) or path in (BASELINE, CHECKPOINT)) and digest(path) != sha:
            raise RuntimeError("fixed starting model/rules changed")
    frozen = source_hashes() | {p: digest(p) for p in (BASELINE, CHECKPOINT)}
    config = {"experiment": "long-timescale-online-v1", "generations": GENERATIONS, "games_per_generation": GAMES,
              "evaluation_seeds_per_generation": 1000, "final_holdout_seeds": 5000,
              "final_candidate": "Gen10 only; no cash-based checkpoint selection", "late_window": [8, 9, 10],
              "success": "all Gen8-10 anchor-minus-baseline CI95 lower>0 AND final holdout lower>0",
              "safety": "two consecutive generations anchor-minus-baseline CI95 upper < -5; holdout unused on early stop",
              "temperature": .5, "anchor_lambda": .3, "reference": old["behavior"],
              "optimizer_loss_protocol": old, "CQL": 0, "loss": "terminal cash/100 MC half MSE only",
              "schedule": "unchanged linear20+cosine300; continuous step; fixed 3e-6 after step320",
              "one_pass": True, "replay": False, "offline_rows": False,
              "sampling_cost": "actual realized training cash, not deployment or paired evaluation",
              "HU_PASS": "unchanged greedy anchored choice", "frozen_sha256": frozen,
              "bootstrap": "10000 seed-cluster resamples; per-generation descriptive, single final holdout primary",
              "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0)}
    atomic_json(root / "config.json", config)
    atomic_json(root / "seed-lock.json", {"generations": {str(g): {"train": seed_specs(g), "evaluation": seed_specs(g, evaluation=True)}
                                                               for g in range(1, 11)}, "final_holdout": seed_specs(holdout=True)})
    parent = load_model(BASELINE, "cuda").requires_grad_(False)
    initial = load_model(CHECKPOINT, "cuda").requires_grad_(False)
    model = load_model(CHECKPOINT, "cuda")
    if parameter_hash(model) != parameter_hash(initial):
        raise RuntimeError("initialization mismatch")
    optimizer = make_optimizer(model)
    rng = np.random.default_rng(20261015)
    updates, results = 0, []
    parent_hash = parameter_hash(parent)
    for generation in range(1, GENERATIONS + 1):
        gen = root / f"gen-{generation:02d}"
        gen.mkdir()
        data_root = gen / "data"
        data_root.mkdir()
        producer = parameter_hash(model)
        actor = AnchoredBoltzmann(model, parent, config["reference"])
        manifest = {"shards": [], "generation": generation, "actor_state_sha256": producer,
                    "producer_checkpoint": CHECKPOINT if generation == 1 else str(root / f"gen-{generation-1:02d}" / "checkpoint.pt")}
        records, train_games = [], []
        specs = seed_specs(generation)
        for start in range(0, GAMES, 300):
            atomic_json(root / "progress.json", {"phase": "collection", "generation": generation, "completed": start, "total": GAMES})
            shard = start // 300
            games, batch, data = collect(actor, parent, specs[start:start+300], gen / f"evidence-{shard:02d}",
                                         f"generation-{generation}", training_rows=True, device="cuda")
            manifest["shards"].append(save_shard(data_root, shard, data))
            records.extend(batch)
            train_games.extend(games)
            atomic_json(data_root / "manifest.json", manifest)
        if parameter_hash(model) != producer:
            raise RuntimeError("collection changed frozen actor")
        atomic_json(gen / "training-outcomes.json", train_games)
        audit = audit_dataset(data_root, manifest)
        rows = load_rows(data_root, manifest)
        behavior = behavior_summary(records)
        detailed = evidence_summary(gen)
        sampling_cash = distribution([g["cash"] for g in train_games])
        del records, data, batch, actor
        before = regression(model, rows, device="cuda")
        q_before = q_probe(model, rows)
        parameters_before = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
        atomic_json(root / "progress.json", {"phase": "MC_update", "generation": generation, "previous_updates": updates})
        updates, losses = update(model, optimizer, rows, rng, updates, device="cuda")
        drift = q_drift(q_before, q_probe(model, rows), rows)
        change = parameter_change(parameters_before, model)
        after = regression(model, rows, device="cuda")
        cp = gen / "checkpoint.pt"
        checkpoint(model, optimizer, cp, updates, rng, generation=generation, producer_state_sha256=producer,
                   manifest_sha256=digest(data_root / "manifest.json"), source_sha256=frozen,
                   initial_checkpoint_sha256=digest(CHECKPOINT), seed_lock_sha256=digest(root / "seed-lock.json"))
        atomic_json(gen / "updates.json", losses)
        training = {"producer_state_sha256": producer, "updated_state_sha256": parameter_hash(model),
                    "checkpoint_sha256": digest(cp), "optimizer_sha256": digest(gen / "optimizer.pt"),
                    "audit": audit, "decisions": len(rows), "updates": updates, "updates_this_generation": len(losses),
                    "behavior": behavior, "diagnostics": detailed, "sampling_execution_cash_unpaired": sampling_cash,
                    "regression_before": before, "regression_after": after, "parameter_change": change, "q_drift": drift}
        atomic_json(gen / "training-result.json", training)
        del rows, q_before, parameters_before, losses
        atomic_json(root / "progress.json", {"phase": "evaluation", "generation": generation})
        evaluation = evaluate(gen / "evaluation", seed_specs(generation, evaluation=True), model, initial, parent,
                              config["reference"], frozen, parent_hash)
        summary = {"generation": generation, "training": training, **evaluation}
        results.append(summary)
        atomic_json(gen / "result.json", summary)
        atomic_json(root / "generations.json", results)
        print(json.dumps({"generation": generation, "cash": summary["cash"], "contrasts": summary["contrasts"]}), flush=True)
        if safety_stop(results):
            break
    completed = len(results) == GENERATIONS
    holdout = None
    if completed:
        # Candidate SHA is locked BEFORE any holdout game. Never select using intermediate cash.
        atomic_json(root / "final-candidate-lock.json", {"generation": 10, "checkpoint": str(cp), "sha256": digest(cp),
                                                        "state_sha256": parameter_hash(model), "seed_lock_sha256": digest(root / "seed-lock.json")})
        atomic_json(root / "progress.json", {"phase": "final_holdout", "generation": 10})
        holdout = evaluate(root / "final-holdout", seed_specs(holdout=True), model, initial, parent,
                           config["reference"], frozen, parent_hash)
    atomic_json(root / "active-checkpoint.json", {"checkpoint": CHECKPOINT, "sha256": digest(CHECKPOINT), "baseline_replaced": False})
    success = completed and all(r["cash"]["anchor"]["seed_bootstrap_95"][0] > 0 for r in results[-3:]) and holdout["cash"]["anchor"]["seed_bootstrap_95"][0] > 0
    atomic_json(root / "result.json", {"status": "complete_stop" if completed else "extreme_safety_stop_rollback",
                                       "generations": results, "holdout": holdout, "stable_improvement": bool(success),
                                       "baseline_replaced": False, "automatic_expansion": False})
    atomic_json(root / "progress.json", {"phase": "complete_stop", "generations": len(results), "stable_improvement": bool(success)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="data/generated/mortal-online-long-v1")
    run(parser.parse_args().directory)
