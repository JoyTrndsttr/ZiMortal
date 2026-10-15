"""Three bounded, synchronous Mortal-style online generations; never promote baseline."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from .active import atomic_json
from .mortal_credit import audit_screening
from .mortal_data import audit_dataset, load_rows, save_shard
from .mortal_deviation_stats import cluster_cohort
from .mortal_extraction import ExtractionPolicy, summarize_families
from .mortal_extraction_experiment import CHECKPOINT
from .mortal_offline import BASELINE, regression
from .mortal_online import checkpoint, collect, make_optimizer, update
from .mortal_sarsa import parameter_hash
from .online_rl import digest, paired_summary, source_hashes
from .runtime import load_model

OFFICIAL = "0cff2b52982be5b1163aa9a62fb01f03ce91e0d2"
GENERATIONS, TRAIN_GAMES, EVAL_SEEDS = 3, 3000, 500


def seed_specs(generation, evaluation=False):
    if evaluation:
        return [(s, s % 3, seat, 0) for s in range(1180000000 + generation * 10000,
                                                               1180000000 + generation * 10000 + EVAL_SEEDS)
                for seat in range(3)]
    return [(1170000000 + generation * 10000 + i, i % 3, (i // 3) % 3, 0)
            for i in range(TRAIN_GAMES)]


def stop_reason(primary, no_improvement, generation):
    _lo, hi = primary["seed_bootstrap_95"]
    if hi < 0:
        return "clear_degradation_rollback"
    if no_improvement >= 3:
        return "three_generations_without_reliable_improvement_rollback"
    if generation >= GENERATIONS:
        return "fixed_generation_budget_stop"
    return None


def assert_frozen(frozen, parent, parent_hash):
    if parameter_hash(parent) != parent_hash or any(digest(p) != sha for p, sha in frozen.items()):
        raise RuntimeError("frozen baseline/source/offline checkpoint changed")


def run(directory):
    torch.set_num_threads(2)
    torch.manual_seed(20261015)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required, no CPU fallback")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    old = json.loads(Path("data/generated/mortal-anchor-30k-v1/config.json").read_text())
    for p, sha in old["frozen_sha256"].items():
        if (p.startswith(("zimortal/engine/", "zimortal/model/")) or p in (BASELINE, CHECKPOINT)) and digest(p) != sha:
            raise RuntimeError("fixed starting rules/model changed")
    frozen = source_hashes() | {p: digest(p) for p in (BASELINE, CHECKPOINT)}
    config = {
        "official_commit": OFFICIAL, "generations": GENERATIONS, "train_games": TRAIN_GAMES,
        "evaluation_seeds_per_generation": EVAL_SEEDS, "frozen_sha256": frozen,
        "behavior": old["behavior"], "anchor_lambda": .3, "epsilon_log": 1e-8,
        "cql": 0., "gamma": 1., "loss": "0.5 MSE chosen Q vs terminal cash/100 only",
        "training_actor": "deterministic anchored argmax; no raw behavior, no extra exploration",
        "training_epochs_per_generation": 1, "batch_size": 256,
        "lr": {"init": 3e-6, "peak": 3e-5, "final": 3e-6, "warmup": 20, "max_steps": 320},
        "optimizer": "new AdamW on offline-to-online transition; persist across generations; decay linear/conv weights only .01",
        "no_offline_data_no_auxiliary_losses": True,
        "stop": "CI95 upper<0, or 3 generations without reliable improvement, or maximum 3",
        "promotion": "CI95 lower>0 AND Bonferroni CI98.333 lower>0; candidate only, no baseline replacement",
        "evaluation_not_used_for_training_or_tuning": True,
        "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0),
    }
    atomic_json(root / "config.json", config)
    atomic_json(root / "seed-lock.json", {str(g): {"train": seed_specs(g), "eval": seed_specs(g, True)} for g in range(1, 4)})
    evidence = root / "official-source"
    evidence.mkdir()
    for path in Path("/tmp/mortal-official-online").glob("*.py"):
        shutil.copyfile(path, evidence / path.name)
    atomic_json(root / "official-source-hashes.json", {str(p.name): digest(p) for p in evidence.glob("*.py")})
    parent = load_model(BASELINE, "cuda").requires_grad_(False).eval()
    initial = load_model(CHECKPOINT, "cuda").requires_grad_(False).eval()
    model = load_model(CHECKPOINT, "cuda")
    initial_hash, parent_hash = parameter_hash(initial), parameter_hash(parent)
    optimizer, rng, updates = make_optimizer(model), np.random.default_rng(20261015), 0
    results, no_improvement, candidates = [], 0, []
    reason = None
    for generation in range(1, GENERATIONS + 1):
        gen = root / f"gen-{generation:02d}"
        gen.mkdir()
        trainroot = gen / "data"
        trainroot.mkdir()
        actor_sha = parameter_hash(model)
        actor = ExtractionPolicy(model, parent, config["behavior"], {"mode": "anchor", "lambda": .3}).eval()
        manifest = {"shards": [], "actor_state_sha256": actor_sha, "generation": generation}
        training_records = []
        specs = seed_specs(generation)
        for start in range(0, len(specs), 300):
            atomic_json(root / "progress.json", {"phase": "on_policy_generation", "generation": generation, "completed": start, "total": TRAIN_GAMES})
            shard = start // 300
            games, records, data = collect(actor, parent, specs[start:start + 300],
                                           gen / f"train-evidence-{shard:02d}", f"generation-{generation}",
                                           training_rows=True, device="cuda")
            manifest["shards"].append(save_shard(trainroot, shard, data))
            training_records.extend(records)
            atomic_json(trainroot / "manifest.json", manifest)
        if parameter_hash(model) != actor_sha:
            raise RuntimeError("generation actor weights/buffers changed during collection")
        data_audit = audit_dataset(trainroot, manifest)
        rows = load_rows(trainroot, manifest)
        before = regression(model, rows, device="cuda")
        atomic_json(root / "progress.json", {"phase": "MC_only_update", "generation": generation, "decisions": len(rows)})
        updates, losses = update(model, optimizer, rows, rng, updates, device="cuda")
        atomic_json(gen / "updates.json", losses)
        cp = gen / "checkpoint.pt"
        checkpoint(model, optimizer, cp, updates, rng, generation=generation,
                   producer_state_sha256=actor_sha, manifest_sha256=digest(trainroot / "manifest.json"),
                   source_sha256=frozen, initial_checkpoint_sha256=digest(CHECKPOINT))
        after = regression(model, rows, device="cuda")
        del rows, actor, data
        policies = {
            "baseline": ExtractionPolicy(None, parent, config["behavior"], {"mode": "baseline"}),
            "initial-anchor": ExtractionPolicy(initial, parent, config["behavior"], {"mode": "anchor", "lambda": .3}),
            "anchor": ExtractionPolicy(model, parent, config["behavior"], {"mode": "anchor", "lambda": .3}),
            "raw": ExtractionPolicy(model, parent, config["behavior"], {"mode": "raw"}),
        }
        evaluation = gen / "evaluation"
        evaluation.mkdir()
        outcomes, metrics = {}, {}
        state_hash = parameter_hash(model)
        for name, policy in policies.items():
            atomic_json(root / "progress.json", {"phase": "independent_evaluation", "generation": generation, "policy": name})
            games, records, _ = collect(policy.eval(), parent, seed_specs(generation, True), evaluation, name, device="cuda")
            outcomes[name] = games
            metrics[name] = summarize_families(records)
            atomic_json(evaluation / "screening-outcomes.json", outcomes)
        atomic_json(root / "progress.json", {"phase": "independent_full_audit", "generation": generation})
        eval_audit = audit_screening(evaluation)
        base = {(g["seed"], g["seat"]): g for g in outcomes["baseline"]}
        startbase = {(g["seed"], g["seat"]): g for g in outcomes["initial-anchor"]}
        cash = {name: cluster_cohort(games, base) for name, games in outcomes.items()}
        change = cluster_cohort(outcomes["anchor"], startbase)
        simultaneous = paired_summary(outcomes["anchor"], outcomes["baseline"], 1515, alpha=.05 / 3)["0"]
        reliable = cash["anchor"]["seed_bootstrap_95"][0] > 0 and simultaneous["paired_seed_bootstrap_interval"][0] > 0
        if reliable:
            candidates.append(str(cp))
            no_improvement = 0
        else:
            no_improvement += 1
        assert_frozen(frozen, parent, parent_hash)
        if parameter_hash(model) != state_hash or parameter_hash(initial) != initial_hash:
            raise RuntimeError("evaluation modified neural weights or BN buffers")
        summary = {
            "generation": generation, "checkpoint": str(cp), "checkpoint_sha256": digest(cp),
            "producer_state_sha256": actor_sha, "updated_state_sha256": state_hash,
            "training_audit": data_audit, "training_decisions": len(training_records),
            "training_behavior": summarize_families(training_records),
            "regression_before": before, "regression_after": after,
            "updates_this_generation": len(losses), "total_updates": updates,
            "cash": cash, "anchor_minus_initial_anchor": change, "family_metrics": metrics,
            "simultaneous_promotion_interval": simultaneous, "reliable_improvement": reliable,
            "evaluation_audit": eval_audit, "baseline_replaced": False,
        }
        results.append(summary)
        atomic_json(gen / "result.json", summary)
        atomic_json(root / "generations.json", results)
        print(json.dumps({"generation": generation, "cash": cash, "promotion": reliable}), flush=True)
        reason = stop_reason(cash["anchor"], no_improvement, generation)
        if reason:
            break
    assert_frozen(frozen, parent, parent_hash)
    rollback = not candidates or "rollback" in reason
    if rollback:
        # A local active pointer reverts; original files and baseline remain untouched.
        atomic_json(root / "active-checkpoint.json", {"checkpoint": CHECKPOINT, "sha256": digest(CHECKPOINT), "reason": reason})
    else:
        atomic_json(root / "active-checkpoint.json", {"checkpoint": candidates[-1], "reason": "experimental candidate only"})
    atomic_json(root / "result.json", {
        "status": "complete_small_online_stop", "stop_reason": reason,
        "generations": results, "candidates": candidates, "rolled_back": rollback,
        "baseline_replaced": False, "training_updates": updates, "automatic_expansion": False,
    })
    atomic_json(root / "progress.json", {"phase": "complete_stop", "generations": len(results), "reason": reason, "rolled_back": rollback})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="data/generated/mortal-online-v1")
    run(parser.parse_args().directory)
