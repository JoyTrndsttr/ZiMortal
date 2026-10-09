"""Fixed-particle A/B performance diagnostic; never produces training labels."""

import argparse
import cProfile
import json
import math
import pstats
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from zimortal.belief.conditional import HistorySampler
from zimortal.engine import ActionType, RuleEngine
from zimortal.engine.evaluator import _decompose

from .active import deserialize_observation
from .adaptive import payoff_bounds
from .rollout import teacher
from .rollout_engine import (
    RolloutEngine,
    _chi_cached,
    _evaluate_cached,
    cached_chi,
    cached_evaluate_hand,
)
from .runtime import load_model


def benchmark(obs, model, particles, seed, directory, name, engine, profile=True):
    _decompose.cache_clear()
    _evaluate_cached.cache_clear()
    _chi_cached.cache_clear()
    profiler = cProfile.Profile()
    started = time.perf_counter()
    with ExitStack() as stack:
        stack.enter_context(patch("zimortal.training.rollout.RuleEngine", engine))
        if engine is RolloutEngine:
            stack.enter_context(patch("zimortal.engine.game.evaluate_hand", cached_evaluate_hand))
            stack.enter_context(
                patch("zimortal.engine.game.enumerate_chi_with_required_bi", cached_chi)
            )
        if profile:
            profiler.enable()
        result = teacher(
            obs,
            model=model,
            rollouts=particles,
            seed=seed,
            batched=True,
            keep_outcomes=True,
            world_sampler=HistorySampler(obs),
        )
        if profile:
            profiler.disable()
    elapsed = time.perf_counter() - started
    if profile:
        profiler.dump_stats(str(directory / f"{name}.prof"))
        with (directory / f"{name}.txt").open("w") as file:
            pstats.Stats(profiler, stream=file).sort_stats("cumulative").print_stats(35)
    return result, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/generated/active-v4")
    parser.add_argument("--output", default="logs/performance/rollout")
    parser.add_argument("--roots", type=int, default=3)
    parser.add_argument("--particles", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root, output = Path(args.data), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((root / "config.json").read_text())
    model = load_model(config["parent_path"])
    # Select different actionable strata deterministically, independently of runtime.
    selected, seen = [], set()
    for path in sorted((root / "candidates").glob("*.json")):
        candidate = json.loads(path.read_text())
        tags = candidate["tags"]
        key = "chi_bi" if "chi_bi" in tags else "peng_pass" if "peng_pass" in tags else "other"
        if key not in seen:
            selected.append((path, candidate))
            seen.add(key)
        if len(selected) >= args.roots:
            break
    rows = []
    for index, (path, candidate) in enumerate(selected):
        obs = deserialize_observation(candidate["observation"])
        baseline, before = benchmark(
            obs, model, args.particles, 880000 + index, output, f"{index}-baseline", RuleEngine
        )
        faster, after = benchmark(
            obs, model, args.particles, 880000 + index, output, f"{index}-cached", RolloutEngine
        )
        timed_fast, after = benchmark(
            obs,
            model,
            args.particles,
            880000 + index,
            output,
            "timed",
            RolloutEngine,
            profile=False,
        )
        timed_original, before = benchmark(
            obs, model, args.particles, 880000 + index, output, "timed", RuleEngine, profile=False
        )
        if timed_fast != timed_original or baseline != timed_original:
            raise RuntimeError("timing repeats changed fixed-particle outcomes")
        if baseline != faster:
            raise RuntimeError("optimization changed fixed-particle outcomes")
        rows.append(
            {
                "root": path.stem,
                "tags": candidate["tags"],
                "actions": len(obs.legal_actions),
                "baseline_seconds": before,
                "cached_seconds": after,
                "speedup": before / after,
                "identical_outcomes": True,
                "payoff_bounds": payoff_bounds(obs),
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    # Diagnose the two terms in the EXACT existing interval, without changing certification.
    intervals = []
    for path in sorted((root / "labels").glob("*.npz")):
        candidate_path = root / "candidates" / f"{path.stem}.json"
        if not candidate_path.exists():
            continue
        candidate = json.loads(candidate_path.read_text())
        obs = deserialize_observation(candidate["observation"])
        metadata = json.loads(path.with_suffix(".json").read_text())
        trace = metadata.get("trace", [])
        if not trace:
            continue
        last = trace[-1]
        low, high = payoff_bounds(obs)
        delta = last["delta_per_pair"]
        n = last["n"]
        with np.load(path, allow_pickle=False) as compact:
            q = compact["q"].astype(float) * 100
            paired_se = compact["paired_se"].astype(float) * 100
        best, reference = int(q.argmax()), candidate["reference"]
        log = math.log(4 / delta)
        pair_range = high - low
        if all(obs.legal_actions[i].kind != ActionType.HU for i in (best, reference)):
            pair_range *= 2
        intervals.append(
            {
                "root": path.stem,
                "tags": candidate["tags"],
                "n": n,
                "resolved": last["resolved"],
                "payoff_range": high - low,
                "best_reference_gap": float(q[best] - q[reference]),
                "best_reference_variance_term": math.sqrt(2 * log) * float(paired_se[best]),
                "best_reference_range_term": 7 * pair_range * log / (3 * (n - 1))
                if best != reference
                else 0,
                "non_hu_range_term": 7 * 2 * (high - low) * math.log(4 / delta) / (3 * (n - 1)),
            }
        )
    report = {
        "roots": rows,
        "intervals": intervals,
        "note": "Wall timings exclude profiling; evaluator cache reset per run. Fixed-particle outcomes identical. No labels generated.",
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
