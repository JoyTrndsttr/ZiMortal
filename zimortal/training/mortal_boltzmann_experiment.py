"""Fixed A/B online exploration pilot: T=.5 only, maximum three generations."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .active import atomic_json
from .mortal_boltzmann import AnchoredBoltzmann
from .mortal_credit import audit_screening
from .mortal_data import audit_dataset, load_rows, save_shard
from .mortal_deviation_stats import cluster_cohort
from .mortal_diagnostics import distribution
from .mortal_extraction import ExtractionPolicy, summarize_families
from .mortal_extraction_experiment import CHECKPOINT
from .mortal_offline import BASELINE, regression
from .mortal_online import checkpoint, collect, make_optimizer, update
from .mortal_online_experiment import assert_frozen
from .mortal_sarsa import parameter_hash
from .online_rl import digest, paired_summary, source_hashes
from .runtime import load_model

GENERATIONS = 3


def seed_specs(generation, evaluation=False):
    start = (1200000000 if evaluation else 1190000000) + generation * 10000
    if evaluation:
        return [(s, s % 3, seat, 0) for s in range(start, start + 500) for seat in range(3)]
    return [(start + i, i % 3, (i // 3) % 3, 0) for i in range(3000)]


def behavior_summary(records):
    result = summarize_families(records)
    for family, summary in result.items():
        rows = [r for r in records if family == "all" or r["family"] == family]
        values = [r["behavior_probability"] for r in rows]
        summary["chosen_p_reference"] = distribution(values)
        summary["p_reference_bins"] = {
            f"[{lo},{hi})": sum(lo <= p < hi for p in values)
            for lo, hi in zip((0., .01, .03, .1, .3, .5), (.01, .03, .1, .3, .5, 1.000001), strict=True)
        }
        summary["actual_chosen_probability"] = distribution([r.get("actual_behavior_probability", 1.) for r in rows])
    return result


def run(directory):
    torch.set_num_threads(2)
    torch.manual_seed(20261015)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    old = json.loads(Path("data/generated/mortal-online-v1/config.json").read_text())
    for path, sha in old["frozen_sha256"].items():
        if (path.startswith(("zimortal/engine/", "zimortal/model/")) or path in (BASELINE, CHECKPOINT)) and digest(path) != sha:
            raise RuntimeError("fixed starting model/rules changed")
    frozen = source_hashes() | {p: digest(p) for p in (BASELINE, CHECKPOINT)}
    config = {"generations": 3, "games_per_arm_generation": 3000, "evaluation_seeds": 500,
              "temperature": .5, "anchor_lambda": .3, "reference": old["behavior"],
              "frozen_sha256": frozen, "optimizer_loss_protocol": old,
              "HU_PASS": "same deterministic anchored choice in both arms, no added pass exploration",
              "training": "A greedy, B full anchored-score Boltzmann except HU/PASS",
              "primary_evaluation": "both learned Q models use greedy anchored extraction",
              "secondary_evaluation": "B sampling actor actual cash; raw A/B diagnostic only",
              "early_stop": "B-greedy CI95 upper<0 against BOTH A-greedy and baseline; otherwise fixed 3 generations",
              "inference": "95% direction signal; also report 99.167% intervals for 3 generations x 2 primary contrasts",
              "no_temperature_or_lambda_selection": True,
              "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0)}
    atomic_json(root / "config.json", config)
    atomic_json(root / "seed-lock.json", {str(g): {"train_A_and_B": seed_specs(g), "evaluation": seed_specs(g, True)} for g in range(1, 4)})
    parent = load_model(BASELINE, "cuda").requires_grad_(False)
    models = {a: load_model(CHECKPOINT, "cuda") for a in ("A", "B")}
    initial_hash = parameter_hash(models["A"])
    if parameter_hash(models["B"]) != initial_hash:
        raise RuntimeError("A/B initialization mismatch")
    optimizers = {a: make_optimizer(m) for a, m in models.items()}
    rngs = {a: np.random.default_rng(20261015) for a in models}
    updates = dict.fromkeys(models, 0)
    parent_hash = parameter_hash(parent)
    results = []
    reason = "fixed_three_generation_stop"
    for generation in range(1, GENERATIONS + 1):
        gen = root / f"gen-{generation:02d}"
        gen.mkdir()
        arms = {}
        for arm, model in models.items():
            armroot = gen / arm
            armroot.mkdir()
            data_root = armroot / "data"
            data_root.mkdir()
            producer = parameter_hash(model)
            actor = (ExtractionPolicy(model, parent, config["reference"], {"mode": "anchor", "lambda": .3})
                     if arm == "A" else AnchoredBoltzmann(model, parent, config["reference"]))
            manifest = {"shards": [], "arm": arm, "generation": generation, "actor_state_sha256": producer}
            records = []
            specs = seed_specs(generation)
            for start in range(0, 3000, 300):
                atomic_json(root / "progress.json", {"phase": "real_on_policy_collection", "generation": generation, "arm": arm, "completed": start, "total": 3000})
                shard = start // 300
                _, batch_records, data = collect(actor, parent, specs[start:start + 300], armroot / f"evidence-{shard:02d}",
                                                  f"{arm}-generation-{generation}", training_rows=True, device="cuda")
                manifest["shards"].append(save_shard(data_root, shard, data))
                records.extend(batch_records)
                atomic_json(data_root / "manifest.json", manifest)
            if parameter_hash(model) != producer:
                raise RuntimeError("collection changed frozen actor")
            audit = audit_dataset(data_root, manifest)
            rows = load_rows(data_root, manifest)
            before = regression(model, rows, device="cuda")
            atomic_json(root / "progress.json", {"phase": "MC_only_update", "generation": generation, "arm": arm})
            previous = updates[arm]
            updates[arm], losses = update(model, optimizers[arm], rows, rngs[arm], previous, device="cuda")
            cp = armroot / "checkpoint.pt"
            checkpoint(model, optimizers[arm], cp, updates[arm], rngs[arm], arm=arm, generation=generation,
                       producer_state_sha256=producer, manifest_sha256=digest(data_root / "manifest.json"),
                       source_sha256=frozen, initial_checkpoint_sha256=digest(CHECKPOINT))
            atomic_json(armroot / "updates.json", losses)
            arms[arm] = {"producer_state_sha256": producer, "updated_state_sha256": parameter_hash(model),
                         "checkpoint": str(cp), "checkpoint_sha256": digest(cp),
                         "training_audit": audit, "training_behavior": behavior_summary(records),
                         "training_decisions": len(rows), "updates": updates[arm], "updates_this_generation": len(losses),
                         "regression_before": before, "regression_after": regression(model, rows, device="cuda")}
            atomic_json(armroot / "training-result.json", arms[arm])
            del rows, data, records, actor
        def greedy(m, mode="anchor"):
            return ExtractionPolicy(m, parent, config["reference"], {"mode": mode, "lambda": .3})
        policies = {"baseline": greedy(None, "baseline"), "A-anchor": greedy(models["A"]),
                    "B-anchor": greedy(models["B"]), "B-sampling": AnchoredBoltzmann(models["B"], parent, config["reference"]),
                    "A-raw": greedy(models["A"], "raw"), "B-raw": greedy(models["B"], "raw")}
        evaluation = gen / "evaluation"
        outcomes, metrics = {}, {}
        state_hashes = {a: parameter_hash(m) for a, m in models.items()}
        for name, policy in policies.items():
            atomic_json(root / "progress.json", {"phase": "independent_evaluation", "generation": generation, "policy": name})
            games, records, _ = collect(policy.eval(), parent, seed_specs(generation, True), evaluation, name, device="cuda")
            outcomes[name] = games
            metrics[name] = behavior_summary(records)
            atomic_json(evaluation / "screening-outcomes.json", outcomes)
        atomic_json(root / "progress.json", {"phase": "full_evaluation_audit", "generation": generation})
        audit = audit_screening(evaluation)
        baseline = {(g["seed"], g["seat"]): g for g in outcomes["baseline"]}
        a_reference = {(g["seed"], g["seat"]): g for g in outcomes["A-anchor"]}
        cash = {name: cluster_cohort(games, baseline) for name, games in outcomes.items()}
        contrasts = {name: cluster_cohort(outcomes[name], a_reference) for name in ("B-anchor", "B-sampling")}
        adjusted = {name: paired_summary(outcomes["B-anchor"], outcomes[name], 1515, alpha=.05 / 6)["0"]
                    for name in ("baseline", "A-anchor")}
        assert_frozen(frozen, parent, parent_hash)
        if any(parameter_hash(m) != state_hashes[a] for a, m in models.items()):
            raise RuntimeError("evaluation changed weights/BN")
        summary = {"generation": generation, "arms": arms, "cash": cash, "B_minus_A": contrasts,
                   "adjusted_primary_intervals": adjusted, "evaluation_behavior": metrics, "evaluation_audit": audit,
                   "direction_signal_95": cash["B-anchor"]["seed_bootstrap_95"][0] > 0 or contrasts["B-anchor"]["seed_bootstrap_95"][0] > 0}
        results.append(summary)
        atomic_json(gen / "result.json", summary)
        atomic_json(root / "generations.json", results)
        print(json.dumps({"generation": generation, "cash": cash, "B_minus_A": contrasts}), flush=True)
        if cash["B-anchor"]["seed_bootstrap_95"][1] < 0 and contrasts["B-anchor"]["seed_bootstrap_95"][1] < 0:
            reason = "clear_B_degradation_against_A_and_baseline_stop"
            break
    atomic_json(root / "active-checkpoint.json", {"checkpoint": CHECKPOINT, "sha256": digest(CHECKPOINT), "note": "experiment stopped; no automatic baseline replacement"})
    atomic_json(root / "result.json", {"status": "complete_online_exploration_AB_stop", "stop_reason": reason,
                                       "generations": results, "initial_state_sha256": initial_hash,
                                       "baseline_replaced": False, "automatic_expansion": False})
    atomic_json(root / "progress.json", {"phase": "complete_stop", "generations": len(results), "reason": reason})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="data/generated/mortal-online-exploration-v1")
    run(parser.parse_args().directory)
