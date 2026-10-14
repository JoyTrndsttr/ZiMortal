"""Fixed mortal-v2 trajectories: MC versus 3/5-step behavior SARSA; then STOP."""

import argparse
import gzip
import json
import time
from pathlib import Path

import torch

from zimortal.model.mortal_q import MortalQNet

from .active import atomic_json
from .mortal_data import audit_dataset, load_rows
from .mortal_diagnostics import select_gate
from .mortal_offline import BASELINE, UtilizationSampler, fit
from .mortal_v2 import game_split
from .online_rl import digest, paired_summary, play, source_hashes
from .runtime import load_model


def improvement(candidate, reference):
    """Predeclared descriptive screening trigger, not a statistical certificate."""
    checks = {}
    for split in ("selected_dev", "heldout"):
        c, m = candidate[split], reference[split]
        deltas = [
            c["per_family"][f]["correlation"] - m["per_family"][f]["correlation"]
            for f in ("chi/pass", "peng/pass")
        ]
        checks[split] = {
            "overall_correlation_delta": c["correlation"] - m["correlation"],
            "rmse_ratio": c["rmse"] / m["rmse"],
            "chi_correlation_delta": deltas[0],
            "peng_correlation_delta": deltas[1],
            "passed": (
                c["correlation"] >= m["correlation"] + 0.03
                and c["rmse"] <= m["rmse"] * 1.005
                and sum(deltas) / 2 >= 0.02
                and min(deltas) >= -0.01
            ),
        }
    return {"passed": all(c["passed"] for c in checks.values()), "checks": checks}


def screening(root, results, baseline, device):
    specs = [
        (seed, seed % 3, seat, 0) for seed in range(1030000000, 1030000500) for seat in range(3)
    ]
    outcomes, failures = {}, []
    for name in ("baseline", "mc", "sarsa-3", "sarsa-5"):
        model = baseline if name == "baseline" else load_model(results[name]["checkpoint"], device)
        atomic_json(
            root / "progress.json",
            {"status": "screening", "policy": name, "independent_seeds": 500, "games": 1500},
        )
        with gzip.open(root / f"screen-{name}-replays.jsonl.gz", "wt") as file:

            def sink(evidence):
                file.write(json.dumps(evidence) + "\n")

            _, games, errors = play(
                model, [baseline], specs, device=device, slots=32, training=False, replay_sink=sink
            )
        outcomes[name] = games
        failures.extend({"policy": name, **e} for e in errors)
    atomic_json(root / "screening-outcomes.json", outcomes)
    if failures:
        atomic_json(root / "screening-failures.json", failures)
        return {"valid": False, "reason": "whole-game failures; no unmatched paired conclusions"}
    audit = audit_screening(root)
    return {
        "audit": audit,
        "valid": True,
        "independent_seeds": 500,
        "games_per_policy": 1500,
        "policy": "raw legal Q greedy, no confidence gate; two frozen baseline opponents",
        "comparisons": {
            name: {
                "vs_baseline": paired_summary(g, outcomes["baseline"], 1414),
                "vs_mc": paired_summary(g, outcomes["mc"], 1414),
            }
            for name, g in outcomes.items()
        },
        "formal_promotion": False,
    }


def audit_screening(root):
    """Independently check every saved step, complete pairs, and actual cash."""
    from zimortal.engine import GameState, RuleEngine

    root = Path(root)
    outcomes = json.loads((root / "screening-outcomes.json").read_text())
    games, steps = 0, 0
    hashes = {}
    for name, results in outcomes.items():
        indexed = {(r["seed"], r["seat"]): r for r in results}
        seen = set()
        path = root / f"screen-{name}-replays.jsonl.gz"
        hashes[path.name] = digest(path)
        with gzip.open(path, "rt") as file:
            for line in file:
                evidence = json.loads(line)
                key = (evidence["seed"], evidence["seat"])
                if key in seen or key not in indexed:
                    raise RuntimeError("duplicate/unmatched screening game")
                engine = RuleEngine()
                end = GameState.deserialize(evidence["state"])
                end.validate()
                state = engine.new_game(evidence["seed"], evidence["dealer"])
                state.validate()
                for action in end.history:
                    state = engine.step(state, action)
                    state.validate()
                    steps += 1
                cash = end.settlement.payments[evidence["seat"]] if end.settlement else 0
                if (
                    not end.terminal
                    or state.serialize() != end.serialize()
                    or cash != indexed[key]["cash"]
                ):
                    raise RuntimeError("screening replay/cash mismatch")
                seen.add(key)
                games += 1
        if seen != indexed.keys():
            raise RuntimeError("incomplete screening replay evidence")
    result = {
        "games": games,
        "steps": steps,
        "per_step_conservation_verified": True,
        "legal_complete_replays_verified": True,
        "cash_labels_verified": True,
        "replay_sha256": hashes,
        "source_hashes": source_hashes(),
    }
    atomic_json(root / "screening-audit.json", result)
    return result


def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(1414)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    root, data = Path(args.directory), Path(args.data)
    root.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((data / "manifest.json").read_text())
    old_config = json.loads((data.parent / "config.json").read_text())
    if digest(BASELINE) != old_config["baseline_sha256"]:
        raise RuntimeError("mortal-v2 baseline changed")
    config = {
        **vars(args),
        "source_hashes": source_hashes(),
        "manifest_sha256": digest(data / "manifest.json"),
        "baseline_sha256": digest(BASELINE),
        "gamma": 1.0,
        "alpha_cql": 0.0,
        "seed": 1414,
        "epochs": 20,
        "patience": 4,
        "batch_size": 256,
        "optimizer": "AdamW lr=3e-4",
        "device": "cuda",
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(0),
        "screening_trigger": "dev and test: corr +.03; RMSE <=1.005xMC; mean chi/peng corr +.02, neither worse by >.01",
    }
    for path, sha in old_config["source_hashes"].items():
        if path.startswith(("zimortal/engine/", "zimortal/model/")) and digest(path) != sha:
            raise RuntimeError("mortal-v2 engine/representation changed")
    atomic_json(root / "config.json", config)
    # Reproduce the v2 initialization RNG path, including baseline construction.
    baseline = load_model(BASELINE, "cuda").requires_grad_(False)
    initial = MortalQNet().cuda()
    results = {}
    with UtilizationSampler() as sampler:
        sampler.phase = "audit"
        begin = time.perf_counter()
        audit = audit_dataset(data, manifest)
        audit["seconds"] = time.perf_counter() - begin
        if audit["games"] != 10000:
            raise RuntimeError("exact existing 10k required")
        atomic_json(root / "audit.json", audit)
        rows = load_rows(data, manifest)
        train, dev, test = game_split(rows)
        sampler.phase = "training"
        for name, n in [("mc", 0), ("sarsa-3", 3), ("sarsa-5", 5)]:
            results[name] = fit(
                initial,
                train,
                dev,
                test,
                root / name,
                device="cuda",
                alpha=0.0,
                epochs=20,
                patience=4,
                gated_diagnostic=True,
                n_step=n,
                target_sync_updates=args.target_sync_updates,
            )
            value = results[name]
            threshold = select_gate(value["selected_dev"])
            value["confidence_gate"] = {
                "threshold": threshold,
                "test_changed_rate": value["heldout"]["gate_rates"][str(threshold)],
            }
            value["heldout"].pop("gate_rates")
            atomic_json(root / name / "result.json", value)
        if len({v["initial_parameter_sha256"] for v in results.values()}) != 1:
            raise RuntimeError("initializations differ")
        common = min(v["epochs_run"] for v in results.values())
        for i in range(common):
            if len({v["epochs"][i]["batch_order_sha256"] for v in results.values()}) != 1:
                raise RuntimeError("batch sequences differ")
        triggers = {
            name: improvement(results[name], results["mc"]) for name in ("sarsa-3", "sarsa-5")
        }
        if any(c["passed"] for c in triggers.values()):
            sampler.phase = "screening"
            screen = screening(root, results, baseline, "cuda")
        else:
            screen = {
                "run": False,
                "reason": "no SARSA variant passed predeclared improvement trigger",
            }
    if (
        source_hashes() != config["source_hashes"]
        or digest(BASELINE) != config["baseline_sha256"]
        or digest(data / "manifest.json") != config["manifest_sha256"]
    ):
        raise RuntimeError("frozen sources/baseline/data manifest changed")
    # Input shards are checked again, without repeating encoding/replay CPU work.
    for shard in manifest["shards"]:
        if (
            digest(data / shard["data"]) != shard["sha256"]
            or digest(data / shard["replays"]) != shard["replay_sha256"]
        ):
            raise RuntimeError("input evidence changed")
    result = {
        "status": "complete_credit_assignment_stop",
        "audit": audit,
        "decision_split": [len(x) for x in (train, dev, test)],
        "variants": results,
        "screening_triggers": triggers,
        "screening": screen,
        "utilization": sampler.samples,
        "baseline_changed": False,
    }
    atomic_json(root / "result.json", result)
    atomic_json(root / "progress.json", {"status": result["status"]})
    print(result["status"], flush=True)


def screen_existing(args):
    """User-authorized exploratory screening; retain failed predeclared gates."""
    torch.set_num_threads(2)
    root = Path(args.directory)
    result = json.loads((root / "result.json").read_text())
    original = json.loads((root / "config.json").read_text())
    if result["status"] != "complete_credit_assignment_stop" or not torch.cuda.is_available():
        raise RuntimeError("completed offline experiment and CUDA required")
    checksums = {BASELINE: digest(BASELINE)} | {
        v["checkpoint"]: digest(v["checkpoint"]) for v in result["variants"].values()
    }
    if checksums[BASELINE] != original["baseline_sha256"]:
        raise RuntimeError("baseline changed")
    config = {
        "source_hashes": source_hashes(),
        "checkpoint_sha256": checksums,
        "reason": "post-hoc exploratory screening of improved held-out chi/peng signal, authorized by user; predeclared joint dev/test gate remains failed",
        "independent_seeds": 500,
        "seed_start": 1030000000,
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(0),
        "formal_promotion": False,
    }
    atomic_json(root / "screening-config.json", config)
    baseline = load_model(BASELINE, "cuda").requires_grad_(False)
    result["screening"] = screening(root, result["variants"], baseline, "cuda")
    result["screening"]["post_hoc_exploratory"] = True
    result["screening"]["predeclared_joint_gate_passed"] = False
    if source_hashes() != config["source_hashes"] or any(
        digest(p) != sha for p, sha in checksums.items()
    ):
        raise RuntimeError("screening sources/checkpoints changed")
    atomic_json(root / "result.json", result)
    atomic_json(root / "progress.json", {"status": "complete_credit_assignment_stop"})
    print("complete_credit_assignment_screening_stop", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="data/generated/mortal-credit-v1")
    parser.add_argument("--data", default="data/generated/mortal-v2/data")
    parser.add_argument("--target-sync-updates", type=int, default=250)
    parser.add_argument("--screen-existing", action="store_true")
    args = parser.parse_args()
    if args.target_sync_updates < 1 or (Path(args.directory).exists() and not args.screen_existing):
        parser.error("positive sync interval and new experiment directory required")
    if args.screen_existing and (
        (Path(args.directory) / "screening-config.json").exists()
        or any(Path(args.directory).glob("screen-*-replays.jsonl.gz"))
    ):
        parser.error("screening evidence already exists; never overwrite")
    try:
        if args.screen_existing:
            screen_existing(args)
        else:
            run(args)
    except Exception as exc:
        if Path(args.directory).exists():
            atomic_json(
                Path(args.directory) / "progress.json", {"status": "failed", "error": str(exc)}
            )
        raise


if __name__ == "__main__":
    main()
