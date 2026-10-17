"""Prepare and run exactly 500 fixed-hidden roots x all legal actions x 32 worlds."""

import argparse
import gzip
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from zimortal.engine import GameState, RuleClarificationRequired, RuleEngine
from zimortal.training.active import atomic_json, serialize_observation
from zimortal.training.counterfactual_rollout import (
    BOOTSTRAP,
    QUOTAS,
    SEED,
    WORLDS,
    assert_frozen,
    continue_root,
    ev_statistics,
    pick_roots,
    policy_bundle,
    ranking,
    root_predictions,
    seeded,
    state_digest,
    summarize_results,
)
from zimortal.training.mortal_credit_diagnostics import blob
from zimortal.training.mortal_data import write_bytes
from zimortal.training.mortal_extraction_experiment import CHECKPOINT
from zimortal.training.mortal_offline import BASELINE
from zimortal.training.mortal_sarsa import parameter_hash
from zimortal.training.online_rl import digest
from zimortal.training.rollout_engine import RolloutEngine
from zimortal.training.runtime import load_model

POOL = Path("data/generated/mortal-credit-diagnostics-v1/LR-B-samples.json")
SOURCE = Path("data/generated/mortal-lr-horizon-v1")
BEHAVIOR = Path("data/generated/mortal-online-v1/config.json")
CHECKPOINTS = {
    "initial": Path(CHECKPOINT),
    "long10": Path("data/generated/mortal-online-long-v1/gen-10/checkpoint.pt"),
    "A10": SOURCE / "gen-10/A/checkpoint.pt",
    "B10": SOURCE / "gen-10/B/checkpoint.pt",
}


def save_gz(path, value):
    return write_bytes(path, gzip.compress(blob(value), mtime=0))


def read_gz(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def frozen_sources():
    paths = sorted(Path("zimortal").glob("**/*.py"))
    paths += [
        Path("scripts/run-counterfactual-ranking-benchmark.py"),
        Path("scripts/audit-counterfactual-ranking-benchmark.py"),
    ]
    return {str(p): digest(p) for p in paths}


def load_bundle(device):
    models = {k: load_model(v, device) for k, v in CHECKPOINTS.items()}
    baseline = load_model(BASELINE, device)
    behavior = json.loads(BEHAVIOR.read_text())["behavior"]
    anchor = policy_bundle(models, baseline, behavior)
    return models, baseline, behavior, anchor


def prepare(directory, device):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    selected = pick_roots(json.loads(POOL.read_text()))
    atomic_json(root / "selected-locators.json", [r["locator"] for r in selected])
    inputs = {str(p): digest(p) for p in [POOL, BEHAVIOR, Path(BASELINE), *CHECKPOINTS.values()]}
    config = {
        "experiment": "counterfactual-ranking-v1",
        "seed": SEED,
        "quotas": QUOTAS,
        "worlds_per_action": WORLDS,
        "bootstrap": BOOTSTRAP,
        "sampling": "stratified outcome-blind RNG selection from earlier uniform5000 LR-B exact-state sample; no replacement",
        "hidden_world": "FIXED original opponent hidden hands; uniform shuffled remaining wall; NOT information-set posterior",
        "posterior_limitation": "existing HistorySampler checks legal reachability but its allocation weights are not certified initial-deal/behavior-conditioned posterior; not used",
        "continuation": "ONE shared initial offline greedy anchor Q+.3log(p_reference+1e-8) for trainee; two frozen baseline opponents",
        "estimand": "E[terminal cash | true hidden hands at root, public root, action, uniform future wall permutation, common INITIAL greedy-anchor continuation]; NOT Q* or information-set Q",
        "comparison_boundary": "all four critics measured against same continuation; NOT each checkpoints own continuation value",
        "clear": "empirical top1 vs top2 paired percentile CI95 lower >0; pointwise post-selection diagnostic only",
        "ties": "top1_accuracy is canonical strict argmax; best_action_hit_rate accepts all tied empirical maxima; pairwise excludes empirical EV ties, predicted ties half credit",
        "no_absolute_calibration": "rawQ cash/100; margins x100 yuan; anchor x100 prior-score equivalent, not cash; no absolute Q-vs-EV claim",
        "model_sha256": {str(p): digest(p) for p in [Path(BASELINE), *CHECKPOINTS.values()]},
        "source_sha256": frozen_sources(),
        "device": device,
        "new_training_games": 0,
        "optimizer_steps": 0,
        "training_labels": False,
    }
    atomic_json(root / "config.json", config)
    models, baseline, behavior, _anchor = load_bundle(device)
    before = {k: parameter_hash(m) for k, m in (models | {"baseline": baseline}).items()}
    grouped = defaultdict(list)
    for i, r in enumerate(selected):
        name, g, shard, ri, seed, seat, step = r["locator"]
        assert name == "LR-B"
        grouped[g, shard].append((i, ri, seed, seat, step))
    entries = []
    for (g, si), requests in sorted(grouped.items()):
        base = SOURCE / f"gen-{g:02d}/B"
        manifest_path = base / "data/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        shard = manifest["shards"][si]
        replay = base / "data" / shard["replays"]
        assert digest(replay) == shard["replay_sha256"]
        decisions = list((base / f"evidence-{si:02d}").glob("decisions-*.jsonl.gz"))
        assert len(decisions) == 1
        for p in (manifest_path, replay, decisions[0]):
            inputs[str(p)] = digest(p)
        with gzip.open(decisions[0], "rt") as f:
            records = [json.loads(line) for line in f]
        games = {(v["seed"], v["seat"]): v for v in read_gz(replay)}
        for index, ri, seed, seat, step in requests:
            record = records[ri]
            assert (record["seed"], record["seat"], record["step"] - 1) == (seed, seat, step)
            game = games[seed, seat]
            terminal = GameState.deserialize(game["state"])
            engine = RuleEngine()
            initial = engine.new_game(seed, game["dealer"])
            state = engine.replay(initial, terminal.history[:step])
            obs = engine.observation(state, seat)
            assert obs.legal_actions[record["chosen"]] == terminal.history[step]
            assert json.loads(blob(obs.legal_actions)) == record["actions"]
            pred, probability, parent = root_predictions(
                state, models, baseline, behavior, device=device
            )
            assert parent == record["parent"]
            np.testing.assert_allclose(probability, record["probabilities"], atol=1e-4, rtol=1e-4)
            identifier = f"root-{index:04d}"
            payload = {
                "id": identifier,
                "family": record["family"],
                "locator": selected[index]["locator"],
                "root": state.serialize(),
                "initial": initial.serialize(),
                "observation": serialize_observation(obs),
                "legal_actions": record["actions"],
                "original_chosen": record["chosen"],
                "baseline_action": parent,
                "p_reference": probability,
                "predictions": pred,
                "recorded_suffix_steps": len(terminal.history) - step,
                "root_sha256": state_digest(state),
            }
            filename = f"{identifier}-locked.json.gz"
            sha = save_gz(root / filename, payload)
            entries.append(
                {
                    "id": identifier,
                    "family": record["family"],
                    "file": filename,
                    "sha256": sha,
                    "locator": payload["locator"],
                    "legal_actions": len(obs.legal_actions),
                    "recorded_suffix_steps": payload["recorded_suffix_steps"],
                }
            )
    entries.sort(key=lambda r: r["id"])
    assert len(entries) == 500
    actions = sum(e["legal_actions"] for e in entries)
    future = WORLDS * sum(e["legal_actions"] * e["recorded_suffix_steps"] for e in entries)
    neural_fraction = 0.20  # predeclared planning heuristic, not measured rollout evidence
    budget = {
        "states": 500,
        "legal_actions": actions,
        "average_legal_actions": actions / 500,
        "rollouts_per_action": WORLDS,
        "unique_rollout_branches": actions * WORLDS,
        "estimated_future_steps": future,
        "step_estimator": "original recorded remaining-history length x action count x32; alternate branches may differ",
        "estimated_gpu_observation_count": round(future * neural_fraction),
        "estimated_trainee_Q_inferences": round(future * neural_fraction / 3),
        "estimated_gpu_model_forward_calls": round(future * neural_fraction * 4 / 3 / 64),
        "batch_estimator": "planning average effective64 decisions/batch; trainee two models, opponents one; actual counted",
        "root_scoring_model_forward_calls": 500 * 5,
        "inference_estimator": "planning assumption20% steps nonforced,1/3 trainee; opponents baseline-only, trainee baseline+Q; actual counts reported",
        "audit_reexec_same_branches": actions * WORLDS,
        "total_executed_branches_including_audit": actions * WORLDS * 2,
        "source_games": len({(e["locator"][4], e["locator"][5]) for e in entries}),
    }
    atomic_json(root / "budget.json", budget)
    atomic_json(root / "locked-state-manifest.json", entries)
    atomic_json(root / "input-sha256.json", inputs)
    atomic_json(
        root / "lock.json",
        {
            "config_sha256": digest(root / "config.json"),
            "manifest_sha256": digest(root / "locked-state-manifest.json"),
            "locators_sha256": digest(root / "selected-locators.json"),
            "budget_sha256": digest(root / "budget.json"),
            "inputs_sha256": digest(root / "input-sha256.json"),
        },
    )
    assert_frozen(models | {"baseline": baseline}, before)
    atomic_json(root / "progress.json", {"status": "prepared_locked_no_rollouts", "roots": 500})
    print(json.dumps(budget, ensure_ascii=False, indent=2), flush=True)


def verify_lock(root):
    config = json.loads((root / "config.json").read_text())
    lock = json.loads((root / "lock.json").read_text())
    for name, field in [
        ("config.json", "config_sha256"),
        ("locked-state-manifest.json", "manifest_sha256"),
        ("selected-locators.json", "locators_sha256"),
        ("budget.json", "budget_sha256"),
        ("input-sha256.json", "inputs_sha256"),
    ]:
        assert digest(root / name) == lock[field], name
    inputs = json.loads((root / "input-sha256.json").read_text())
    for p, sha in (inputs | config["source_sha256"]).items():
        assert digest(p) == sha, p
    entries = json.loads((root / "locked-state-manifest.json").read_text())
    for e in entries:
        assert digest(root / e["file"]) == e["sha256"]
    return config, entries


def analyze(root, *, save=True):
    results = []
    failures = []
    for e in json.loads((root / "locked-state-manifest.json").read_text()):
        r = read_gz(root / f"{e['id']}-rollouts.json.gz")
        if "failure" in r:
            failures.append({"id": e["id"], "family": e["family"], "error": r["failure"]})
            continue
        results.append(r["result"])
    report = {
        "status": "complete_diagnostic_stop",
        "states": len(results),
        "failures": failures,
        "ranking": summarize_results(results),
        "clear_states": sum(r["ev"]["clear"] for r in results),
        "clear_by_family": {
            f: sum(r["family"] == f and r["ev"]["clear"] for r in results) for f in QUOTAS
        },
        "ties": sum(len(r["ev"]["best_set"]) > 1 for r in results),
        "new_training_games": 0,
        "optimizer_steps": 0,
        "not_for_training": True,
    }
    # Paired game-cluster bootstrap: root sampling uncertainty, NOT nested MC-world uncertainty.
    groups = sorted({r["locator"][4] for r in results})
    rng = np.random.default_rng(SEED)
    samples = rng.integers(len(groups), size=(10000, len(groups)))
    report["paired_checkpoint_comparisons"] = {}
    for mode in ("raw", "anchor"):
        for cp in ("long10", "A10", "B10"):
            out = {}
            for metric in ("best_action_hit_rate", "pairwise_accuracy", "regret_yuan"):
                sums = []
                counts = []
                for g in groups:
                    delta = [
                        r["ranking"][f"{cp}/{mode}"][metric]
                        - r["ranking"][f"initial/{mode}"][metric]
                        for r in results
                        if r["locator"][4] == g
                        and r["ranking"][f"{cp}/{mode}"][metric] is not None
                        and r["ranking"][f"initial/{mode}"][metric] is not None
                    ]
                    sums.append(sum(delta))
                    counts.append(len(delta))
                sums, counts = np.array(sums), np.array(counts)
                valid = counts[samples].sum(1) > 0
                means = sums[samples][valid].sum(1) / counts[samples][valid].sum(1)
                out[metric] = {
                    "delta": float(sums.sum() / counts.sum()) if counts.sum() else None,
                    "seed_cluster_ci95": np.quantile(means, [0.025, 0.975]).tolist()
                    if len(means)
                    else None,
                }
            report["paired_checkpoint_comparisons"][f"{cp}-{mode}-minus-initial"] = out
    if save:
        atomic_json(root / "result.json", report)
    return report


def run(directory, device):
    root = Path(directory)
    _config, entries = verify_lock(root)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    models, baseline, _behavior, anchor = load_bundle(device)
    before = {k: parameter_hash(m) for k, m in (models | {"baseline": baseline}).items()}
    counts = defaultdict(int)
    started = time.perf_counter()
    for i, e in enumerate(entries):
        filename = root / f"{e['id']}-rollouts.json.gz"
        if filename.exists():
            raise RuntimeError("refusing to overwrite existing rollout results")
        payload = read_gz(root / e["file"])
        state = GameState.deserialize(payload["root"])
        try:
            values, evidence = continue_root(
                state, e["id"], anchor, baseline, engine=RolloutEngine(), device=device
            )
            ev = ev_statistics(values, seeded(e["id"], "statistics"))
            strategies = {"baseline": payload["predictions"]["baseline"]}
            for cp in CHECKPOINTS:
                strategies[f"{cp}/raw"] = payload["predictions"][cp]["q"]
                strategies[f"{cp}/anchor"] = payload["predictions"][cp]["anchor"]
            result = {
                "id": e["id"],
                "locator": payload["locator"],
                "family": e["family"],
                "ev": ev,
                "ranking": {k: ranking(v, ev, values) for k, v in strategies.items()},
            }
            save_gz(filename, {"outcomes": values.tolist(), "evidence": evidence, "result": result})
            for key, n in evidence["counts"].items():
                counts[key] += n
        except RuleClarificationRequired as exc:
            save_gz(
                filename, {"failure": str(exc), "whole_root_excluded": True, "no_replacement": True}
            )
        atomic_json(
            root / "progress.json",
            {
                "status": "diagnostic_rollouts",
                "roots_done": i + 1,
                "roots_total": 500,
                "seconds": time.perf_counter() - started,
                "counts": dict(counts),
            },
        )
    assert_frozen(models | {"baseline": baseline}, before)
    verify_lock(root)
    result = analyze(root)
    files = {
        str(root / f"{e['id']}-rollouts.json.gz"): digest(root / f"{e['id']}-rollouts.json.gz")
        for e in entries
    }
    atomic_json(root / "output-sha256.json", files)
    atomic_json(
        root / "run-summary.json",
        {
            "seconds": time.perf_counter() - started,
            "counts": dict(counts),
            "checkpoints_BN_unchanged": True,
            "training_updates": 0,
        },
    )
    atomic_json(root / "progress.json", {"status": result["status"], "roots_done": 500})
    print(result["status"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="data/generated/counterfactual-ranking-v1")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if args.prepare:
        prepare(args.directory, args.device)
    else:
        run(args.directory, args.device)


if __name__ == "__main__":
    main()
