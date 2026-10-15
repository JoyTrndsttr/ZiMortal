"""Finite behavior-coverage A/B: fixed MC+CQL, original support, no online updates."""

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from zimortal.model.mortal_faithful import MortalFaithfulQNet

from .active import atomic_json
from .mortal_coverage_sampling import enhance_probability
from .mortal_data import audit_dataset, generate_dataset, load_rows
from .mortal_data_volume import select_common_update, train_fixed
from .mortal_faithful_diagnostics import coverage
from .mortal_offline import BASELINE, regression
from .mortal_support_experiment import evaluate, offline_support
from .online_rl import digest, paired_summary, source_hashes
from .runtime import load_model


def coverage_evidence(rows):
    result = {}
    for family in ("discard", "chi/pass", "peng/pass", "hu/pass"):
        group = [r for r in rows if r["family"] == family]
        chosen = Counter(str(r["chosen_kind"]) for r in group)
        non_greedy = [r for r in group if int(r["chosen"]) != int(r["parent_greedy"])]
        actual_templates = Counter()
        states = {}
        for r in group:
            action = r["actions"][int(r["chosen"])].tobytes()
            actual_templates[hashlib.sha256(action).hexdigest()] += 1
            key = hashlib.sha256(r["features"].tobytes() + r["actions"].tobytes()).hexdigest()
            states.setdefault(key, set()).add(int(r["chosen"]))
        result[family] = {
            "decisions": len(group),
            "chosen_kinds": dict(chosen),
            "non_greedy_executed": len(non_greedy),
            "non_greedy_rate": len(non_greedy) / max(1, len(group)),
            "original_low_support_executed": sum(
                int(r["reference_probabilities"][int(r["chosen"])] <= .1) for r in group
            ),
            "mean_alternative_probability_mass": float(np.mean([
                1 - r["behavior_probabilities"][int(r["parent_greedy"])] for r in group
            ])) if group else None,
            "distinct_observation_actionsets": len(states),
            "repeated_states_with_multiple_executed_actions": sum(len(v) > 1 for v in states.values()),
            "distinct_executed_action_templates": len(actual_templates),
            "note": "template coverage is not conditional state-action coverage; no counterfactual labels",
        }
    return result


def verify_reference_and_transform(rows, config):
    maximum = 0.
    for r in rows:
        p = r["reference_probabilities"]
        expected = enhance_probability(p, int(r["parent_greedy"]), str(r["family"]), config)
        error = float(np.max(np.abs(expected - r["behavior_probabilities"])))
        maximum = max(maximum, error)
        if error > 1e-6:
            raise RuntimeError("stored exploration transform mismatch")
    return {"decisions": len(rows), "maximum_transform_error": maximum}


def run(directory, reuse_data=False):
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=reuse_data)
    saved_audits = {}
    collection_sha = None
    if reuse_data:
        if any(root.glob("*/fit")) or (root / "result.json").exists():
            raise RuntimeError("reuse only before any training/test evaluation")
        prior = json.loads((root / "config.json").read_text())
        for path, sha in prior["source_hashes"].items():
            sampling_file = path.startswith(("zimortal/engine/", "zimortal/model/")) or path in (
                "zimortal/training/mortal_data.py", "zimortal/training/mortal_coverage_sampling.py",
                "zimortal/training/rollout_engine.py",
            )
            if sampling_file and digest(path) != sha:
                raise RuntimeError("collection representation/sampling changed")
        shutil.copyfile(root / "config.json", root / "collection-config.json")
        collection_sha = digest(root / "collection-config.json")
        saved_audits = json.loads((root / "audits.json").read_text())
    original = json.loads(Path("data/generated/mortal-v2/config.json").read_text())
    if digest(BASELINE) != original["baseline_sha256"]:
        raise RuntimeError("baseline changed")
    previous = json.loads(Path("data/generated/mortal-volume20k-v2/config.json").read_text())
    for path, sha in previous["source_hashes"].items():
        if path.startswith(("zimortal/engine/", "zimortal/model/")) and digest(path) != sha:
            raise RuntimeError("engine/model changed")
    base = original["family_config"] | {"record_reference": True}
    enhanced = base | {"coverage": {"mixture": .5, "minimum": .03, "alternatives": 3}}
    config = {
        "arms": {"A-original": base, "B-coverage": enhanced},
        "training_start": 1100000000, "training_games_per_arm": 4000,
        "heldout_start": 1110000000, "heldout_games": 2000,
        "cash_start": 1120000000, "cash_seeds": 500,
        "updates": 5200, "dev_interval": 260, "batch_size": 256,
        "optimizer": previous["optimizer"], "learning_rate": 3e-4,
        "alpha": .1, "support": .1, "target": "terminal MC cash/100",
        "model": "unchanged faithful128x12 latent1024",
        "source_hashes": source_hashes(), "baseline_sha256": digest(BASELINE),
        "selection": "common update minimizes mean A/B shared new-dev RMSE, earliest tie",
        "support_reference": "original exploration for BOTH arms; no mechanical relabeling",
        "regression_estimand": "shared original-behavior terminal return; B training returns follow B",
        "device": "cuda", "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0),
        "no_online": True, "no50k": True, "reused_collection_config_sha256": collection_sha,
    }
    atomic_json(root / "config.json", config)
    baseline = load_model(BASELINE, "cuda").requires_grad_(False)
    train = {}
    manifests = {}
    audits = {}
    for name, settings in {**config["arms"], "heldout": base}.items():
        heldout = name == "heldout"
        data = root / name / "data"
        if reuse_data:
            manifest = json.loads((data / "manifest.json").read_text())
            if manifest["metadata"]["arms"] != config["arms"]:
                raise RuntimeError("collection exploration differs")
            for shard in manifest["shards"]:
                if digest(data / shard["data"]) != shard["sha256"] or digest(data / shard["replays"]) != shard["replay_sha256"]:
                    raise RuntimeError("collection shard changed")
            audits[name] = saved_audits[name]
            if audits[name]["games"] != (2000 if heldout else 4000) or not audits[name]["replay_verified"]:
                raise RuntimeError("collection incomplete or unaudited")
        else:
            manifest = generate_dataset(
                baseline, data, games=2000 if heldout else 4000,
                start=1110000000 if heldout else 1100000000,
                shard_games=250, device="cuda", slots=32,
                family_config=settings, metadata=config | {"arm": name},
            )
            audits[name] = audit_dataset(data, manifest)
        if manifest["failures"]:
            raise RuntimeError("no exclusions/replacements permitted")
        rows = load_rows(data, manifest)
        audits[name]["probabilities"] = verify_reference_and_transform(rows, settings)
        manifests[name] = digest(data / "manifest.json")
        atomic_json(root / "audits.json", audits)
        if heldout:
            dev = [r for r in rows if int(r["seed"]) % 2]
            test = [r for r in rows if not int(r["seed"]) % 2]
        else:
            train[name] = rows
    if {int(r["seed"]) for r in dev} & {int(r["seed"]) for r in test}:
        raise RuntimeError("heldout overlap")
    atomic_json(root / "training-coverage.json", {n: coverage(r) for n, r in train.items()})
    atomic_json(root / "executed-coverage.json", {n: coverage_evidence(r) for n, r in train.items()})
    torch.manual_seed(1414)
    initial = MortalFaithfulQNet().cuda()
    fits = {n: train_fixed(initial, rows, dev, root / n / "fit") for n, rows in train.items()}
    if len({f["initial_parameter_sha256"] for f in fits.values()}) != 1:
        raise RuntimeError("initialization mismatch")
    common = select_common_update(fits)
    lock = {"common_update": common, "checkpoints": {}}
    models = {}
    for n in train:
        fitroot = root / n / "fit"
        shutil.copyfile(fitroot / f"update-{common:04d}.pt", fitroot / "selected-common.pt")
        shutil.copyfile(fitroot / f"update-{common:04d}-optimizer.pt", fitroot / "selected-common-optimizer.pt")
        lock["checkpoints"][n] = digest(fitroot / "selected-common.pt")
        models[n] = load_model(fitroot / "selected-common.pt", "cuda")
    atomic_json(root / "selection-lock.json", lock)
    offline = {n: {
        "terminal_test": regression(m, test, device="cuda"),
        "raw": offline_support(m, test, None),
        "guarded": offline_support(m, test, .1),
    } for n, m in models.items()}
    atomic_json(root / "offline-test.json", offline)
    atomic_json(root / "heldout-coverage.json", {"dev": coverage(dev), "test": coverage(test)})
    specs = [(s, s % 3, seat, 0) for s in range(1120000000, 1120000500) for seat in range(3)]
    policies = {"baseline": (None, None),
                **{n + "-raw": (m, None) for n, m in models.items()},
                **{n + "-guarded": (m, .1) for n, m in models.items()}}
    screening, outcomes = evaluate(root / "screening", policies, baseline, base, specs)
    direct = {mode: paired_summary(outcomes["B-coverage-" + mode], outcomes["A-original-" + mode], 1414)
              for mode in ("raw", "guarded")}
    reliance = {n: paired_summary(outcomes[n + "-guarded"], outcomes[n + "-raw"], 1414) for n in train}
    for p, sha in config["source_hashes"].items():
        if digest(p) != sha:
            raise RuntimeError("runtime source changed")
    for n, sha in manifests.items():
        if digest(root / n / "data" / "manifest.json") != sha:
            raise RuntimeError("dataset changed")
    atomic_json(root / "result.json", {
        "status": "complete_coverage_stop", "fits": fits, "offline": offline,
        "screening": screening, "B_minus_A": direct, "guard_minus_raw": reliance,
        "online_started": False, "baseline_changed": False,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="data/generated/mortal-coverage-v1")
    parser.add_argument("--reuse-data", action="store_true")
    args = parser.parse_args()
    run(args.directory, args.reuse_data)
