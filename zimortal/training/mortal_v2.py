"""Finite exploration pilot followed by independent 10k MC/CQL; always stop."""

import argparse
import cProfile
import gzip
import io
import json
import pstats
import time
from pathlib import Path

import torch

from zimortal.model.mortal_q import MortalQNet

from .active import atomic_json
from .mortal_data import audit_dataset, generate_dataset, load_rows
from .mortal_diagnostics import coverage, select_gate
from .mortal_offline import BASELINE, UtilizationSampler, fit
from .online_rl import digest, source_hashes
from .runtime import load_model


def game_split(rows):
    result = [
        [
            r
            for r in rows
            if (int(r["game_id"]) % 10 >= 2 if part == 0 else int(r["game_id"]) % 10 == part % 2)
        ]
        for part in range(3)
    ]
    sets = [{int(r["game_id"]) for r in group} for group in result]
    if not all(sets) or any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("invalid complete-game splits")
    return result


def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(1414)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    root = Path(args.directory)
    root.mkdir(parents=True, exist_ok=False)
    family = {
        "discard": args.temperature_discard,
        "chi/pass": args.temperature_chi,
        "peng/pass": args.temperature_peng,
        "hu/pass": args.temperature_hu,
        "hu_pass_probability": args.hu_pass_probability,
    }
    config = {
        **vars(args),
        "family_config": family,
        "source_hashes": source_hashes(),
        "baseline_sha256": digest(BASELINE),
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
    }
    atomic_json(root / "config.json", config)
    baseline = load_model(BASELINE, "cuda").requires_grad_(False)
    model = MortalQNet().cuda()
    stages = {}
    with UtilizationSampler() as sampler:
        for stage, count, start in [
            ("pilot", args.pilot_games, 1010000000),
            ("data", 10000, 1020000000),
        ]:
            if (
                source_hashes() != config["source_hashes"]
                or digest(BASELINE) != config["baseline_sha256"]
            ):
                raise RuntimeError("frozen producer changed")
            sampler.phase = stage
            begin = time.perf_counter()
            profiler = cProfile.Profile() if stage == "pilot" else None
            if profiler:
                profiler.enable()
            manifest = generate_dataset(
                baseline,
                root / stage,
                games=count,
                start=start,
                device="cuda",
                metadata=config,
                family_config=family,
            )
            if profiler:
                profiler.disable()
                stream = io.StringIO()
                pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(
                    "encoding", 40
                )
                (root / "encoding-profile.txt").write_text(stream.getvalue())
            data_seconds = time.perf_counter() - begin
            begin = time.perf_counter()
            atomic_json(
                root / "progress.json", {"status": "deep_audit", "stage": stage, "games": count}
            )
            audit = audit_dataset(root / stage, manifest)
            audit["seconds"] = time.perf_counter() - begin
            rows = load_rows(root / stage, manifest)
            game_cash = {}
            for shard in manifest["shards"]:
                for game in json.loads(
                    gzip.decompress((root / stage / shard["replays"]).read_bytes())
                ):
                    terminal = json.loads(game["state"])
                    game_cash[game["seed"]] = (
                        terminal["settlement"]["payments"][game["seat"]] / 100
                        if terminal["settlement"]
                        else 0
                    )
            report = {
                "coverage": coverage(rows, game_cash=game_cash),
                "audit": audit,
                "data_seconds": data_seconds,
                "profiles": manifest["profiles"],
                "excluded": len(manifest["failures"]),
            }
            stages[stage] = report
            atomic_json(root / f"{stage}-report.json", report)
            if stage == "pilot":
                f = report["coverage"]["families"]
                passes = report["coverage"]["hu_pass"]["decisions"]
                if not (
                    0.05 <= f["peng/pass"]["changed_rate"] <= 0.25
                    and f["discard"]["changed_rate"] <= 0.30
                    and f["chi/pass"]["changed_rate"] <= 0.30
                    and 0 < passes <= 0.05 * f["hu/pass"]["decisions"]
                ):
                    raise RuntimeError(
                        "pilot coverage outside conservative bounds; inspect before 10k"
                    )
                print(json.dumps({"pilot_passed": report["coverage"]}), flush=True)
        train, dev, test = game_split(rows)
        sampler.phase = "training"
        variants = {}
        for name, alpha in [("mc", 0.0), ("mc-cql", 0.1)]:
            variants[name] = fit(
                model,
                train,
                dev,
                test,
                root / name,
                device="cuda",
                alpha=alpha,
                epochs=20,
                patience=4,
                gated_diagnostic=True,
            )
            value = variants[name]
            threshold = select_gate(value["selected_dev"])
            # Test thresholds are not searched; retain only the dev-selected result.
            value["confidence_gate"] = {
                "selected_on": "dev change budget <=10%",
                "threshold": threshold,
                "test_changed_rate": value["heldout"]["gate_rates"][str(threshold)],
            }
            value["heldout"].pop("gate_rates")
            atomic_json(root / name / "result.json", value)
    if source_hashes() != config["source_hashes"] or digest(BASELINE) != config["baseline_sha256"]:
        raise RuntimeError("producer changed during experiment")
    all_seeds = [seed for shard in manifest["shards"] for seed in shard["seeds"]]
    result = {
        "status": "complete_10k_stop",
        "stages": stages,
        "variants": variants,
        "game_split": [
            sum(s % 10 >= 2 for s in all_seeds),
            sum(s % 10 == 1 for s in all_seeds),
            sum(s % 10 == 0 for s in all_seeds),
        ],
        "decision_game_split": [len({int(r["game_id"]) for r in x}) for x in (train, dev, test)],
        "decision_split": [len(x) for x in (train, dev, test)],
        "parameters": sum(p.numel() for p in model.parameters()),
        "utilization": sampler.samples,
        "baseline_changed": False,
        "cash_strength_evaluated": False,
    }
    atomic_json(root / "result.json", result)
    atomic_json(root / "progress.json", {"status": "complete_10k_stop"})
    print("complete_10k_stop", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="data/generated/mortal-v2")
    parser.add_argument("--pilot-games", type=int, default=1500)
    parser.add_argument("--temperature-peng", type=float, default=2.5)
    parser.add_argument("--temperature-discard", type=float, default=1.5)
    parser.add_argument("--temperature-chi", type=float, default=1.5)
    parser.add_argument("--temperature-hu", type=float, default=1.0)
    parser.add_argument("--hu-pass-probability", type=float, default=0.02)
    args = parser.parse_args()
    if not 1000 <= args.pilot_games <= 2000:
        parser.error("pilot must contain 1000-2000 games")
    import math

    if any(
        not math.isfinite(t) or t <= 0
        for t in (
            args.temperature_discard,
            args.temperature_chi,
            args.temperature_peng,
            args.temperature_hu,
        )
    ):
        parser.error("temperatures must be finite and positive")
    if not 0.01 <= args.hu_pass_probability <= 0.05:
        parser.error("HU/PASS exploration must remain within 1%-5%")
    if Path(args.directory).exists():
        parser.error("experiment directory must be new; existing evidence is preserved")
    try:
        run(args)
    except Exception as exc:
        if Path(args.directory).exists():
            atomic_json(
                Path(args.directory) / "progress.json", {"status": "failed", "error": str(exc)}
            )
        raise


if __name__ == "__main__":
    main()
