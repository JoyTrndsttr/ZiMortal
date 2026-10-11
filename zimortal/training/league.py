"""Continuous cash PPO stages with fresh promotion screens and final audit."""

import argparse
import json
import math
from itertools import pairwise
from pathlib import Path

import torch

from .active import atomic_json
from .cashq import digest
from .online_rl import commit_report, evaluation, run, source_hashes
from .runtime import load_model


def promotion(evaluation_result, tolerance):
    """Approximate conservative research gate, not formal global certification."""
    if not evaluation_result.get("valid"):
        return {"accepted": False, "reason": "invalid games or undefined rules"}
    aggregate = evaluation_result["aggregate"]["paired_seed_bootstrap_interval"][0]
    head = evaluation_result["opponents"]["0"]["paired_seed_bootstrap_interval"][0]
    worst = min(r["mean_cash_delta"] for r in evaluation_result["opponents"].values())
    accepted = bool(aggregate > 0 and head > 0 and worst >= -tolerance)
    return {
        "accepted": accepted,
        "aggregate_lower": aggregate,
        "parent_lower": head,
        "worst_opponent_mean": worst,
        "tolerance": tolerance,
        "reason": "passed frozen experimental cash gate"
        if accepted
        else "insufficient improvement evidence",
        "limitation": "bootstrap is approximate; per-opponent regression guard uses means",
    }


def stage_args(args, stage, parent, initialize, previous_optimizer, best_optimizer, promoted_pool):
    # Never reuse any earlier stage's training, development, or screen seeds.
    historical = args.historical
    fixed = list(dict.fromkeys([args.initial, args.anchor, *args.opponents, *promoted_pool]))
    fixed = [p for p in fixed if p not in (parent, historical)]
    games = args.games if stage == 0 else args.later_games
    return argparse.Namespace(
        base=parent,
        initialize=initialize,
        optimizer_checkpoint=previous_optimizer,
        optimizer_best=best_optimizer,
        holdout_alpha=0.05 / args.stages,
        historical=historical,
        opponents=fixed,
        snapshot_pool=args.snapshot_pool,
        selfplay_fraction=getattr(args, "selfplay_fraction", 0.5),
        value_coef=getattr(args, "value_coef", 0.5),
        entropy_coef=getattr(args, "entropy_coef", 0.01),
        anchor_coef=getattr(args, "anchor_coef", 0.02),
        sampling_temperature=getattr(args, "sampling_temperature", 1.0),
        audit_decisions=True,
        directory=str(Path(args.directory) / f"stage-{stage + 1:02d}"),
        reports=str(Path(args.reports) / f"stage-{stage + 1:02d}"),
        device=args.device,
        iterations=args.generations,
        games=games,
        slots=args.slots,
        epochs=2,
        batch_size=512,
        lr=args.lr,
        seed=args.seed + stage * 100,
        train_start=args.train_start + stage * args.seed_stride,
        dev_start=args.dev_start + stage * args.seed_stride,
        dev_seeds=args.dev_seeds,
        holdout_start=args.screen_start + stage * args.seed_stride,
        holdout_seeds=args.screen_seeds,
        commit_reports=args.commit_reports,
    )


def league(args):
    torch.set_num_threads(2)
    root, reports = Path(args.directory), Path(args.reports)
    root.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    frozen_paths = list(
        dict.fromkeys([args.initial, args.anchor, args.historical, *args.opponents])
    )
    config = {k: v for k, v in vars(args).items() if k != "commit_reports"}
    config.update(
        sources=source_hashes(),
        weights={p: digest(p) for p in frozen_paths},
        initial_optimizer_sha256=digest(args.initial_optimizer) if args.initial_optimizer else None,
    )
    config_path = root / "league-config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("league source/config/initial weights changed; start a new version")
    atomic_json(config_path, config)
    parent, initialize = args.initial, args.initial
    previous_optimizer, best_optimizer = args.initial_optimizer, True
    promoted_pool, stage_reports = [], []
    for stage in range(args.stages):
        if source_hashes() != config["sources"] or any(
            digest(p) != s for p, s in config["weights"].items()
        ):
            raise ValueError("frozen league producer or historical weights changed")
        settings = stage_args(
            args, stage, parent, initialize, previous_optimizer, best_optimizer, promoted_pool
        )
        stage_report_path = reports / f"stage-{stage + 1:02d}-selection.json"
        if not stage_report_path.exists():
            atomic_json(
                root / "league-progress.json",
                {
                    "status": "training",
                    "stage": stage + 1,
                    "stages": args.stages,
                    "training_parent": parent,
                    "initialize": initialize,
                    "stage_progress": str(Path(settings.directory) / "progress.json"),
                    "completed_training_games": sum(r["training_games"] for r in stage_reports),
                    "planned_training_games": args.generations
                    * (args.games + (args.stages - 1) * args.later_games),
                    "promoted_models": promoted_pool,
                },
            )
            run(settings)
            holdout = json.loads((Path(settings.reports) / "holdout.json").read_text())
            decision = promotion(holdout["evaluation"], args.regression_tolerance)
            candidate = str(Path(settings.directory) / "best-dev.pt")
            generations = [
                json.loads(p.read_text())
                for p in sorted(Path(settings.reports).glob("generation-*.json"))
            ]
            result = {
                "stage": stage + 1,
                "parent": parent,
                "parent_sha256": digest(parent),
                "candidate": candidate,
                "candidate_sha256": digest(candidate),
                "promotion": decision,
                "screen_report": str(Path(settings.reports) / "holdout.json"),
                "training_games": sum(r["completed_games"] for r in generations),
                "training_decisions": sum(r["decisions"] for r in generations),
                "excluded_rule_games": sum(r["excluded_rule_games"] for r in generations),
                "next_initialize": candidate
                if decision["accepted"]
                else str(Path(settings.directory) / f"generation-{args.generations:03d}.pt"),
                "optimizer_checkpoint": str(Path(settings.directory) / "last.pt"),
                "formal_champion_changed": False,
            }
            atomic_json(stage_report_path, result)
            if args.commit_reports:
                commit_report(
                    stage_report_path,
                    stage + 1,
                    f"train(league): 记录第{stage + 1}阶段模型晋级检查",
                )
        result = json.loads(stage_report_path.read_text())
        if (
            digest(result["candidate"]) != result["candidate_sha256"]
            or result["parent"] != parent
            or digest(parent) != result["parent_sha256"]
        ):
            raise ValueError("saved league selection chain changed")
        if result["promotion"]["accepted"]:
            parent = result["candidate"]
            promoted_pool.append(parent)
        initialize = result["next_initialize"]
        previous_optimizer = result["optimizer_checkpoint"]
        best_optimizer = result["promotion"]["accepted"]
        stage_reports.append(result)
    final_path = reports / "final-audit.json"
    if not final_path.exists():
        atomic_json(
            root / "league-progress.json",
            {
                "status": "final_audit",
                "training_parent": parent,
                "completed_training_games": sum(r["training_games"] for r in stage_reports),
            },
        )
        models = [load_model(p, args.device) for p in frozen_paths if p != args.initial]
        # Fixed historical lineup; include initial experimental model separately.
        final_paths = [p for p in frozen_paths if p != args.initial] + [args.initial]
        models.append(load_model(args.initial, args.device))
        result = evaluation(
            load_model(parent, args.device),
            load_model(args.anchor, args.device),
            models,
            args.final_start,
            args.final_seeds,
            args.device,
            args.slots,
            audit=True,
        )
        atomic_json(
            final_path,
            {
                "checkpoint": parent,
                "sha256": digest(parent),
                "compared_with": args.anchor,
                "opponent_pool": final_paths,
                "evaluation": result,
                "selection_used_final_audit": False,
                "formal_champion_changed": False,
            },
        )
        if args.commit_reports:
            commit_report(final_path, args.stages, "train(league): 记录连续强化学习最终独立评估")
    atomic_json(
        root / "league-progress.json",
        {
            "status": "complete",
            "stages": args.stages,
            "training_parent": parent,
            "promoted_models": promoted_pool,
            "completed_training_games": sum(r["training_games"] for r in stage_reports),
            "completed_training_decisions": sum(r["training_decisions"] for r in stage_reports),
            "final_audit": str(final_path),
            "formal_champion_changed": False,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", default="data/generated/online-rl-v2/best-dev.pt")
    parser.add_argument("--initial-optimizer", default="data/generated/online-rl-v2/last.pt")
    parser.add_argument("--anchor", default="checkpoints/huxi-resnet.pt")
    parser.add_argument("--historical", default="checkpoints/scale100k-resnet.pt")
    parser.add_argument(
        "--opponents",
        nargs="*",
        default=[
            "checkpoints/boundary-calibrated.pt",
            "checkpoints/planning-cuda-resnet.pt",
            "checkpoints/round3-resnet.pt",
            "data/generated/online-rl-v1/generation-002.pt",
            "data/generated/online-rl-v1/generation-012.pt",
        ],
    )
    parser.add_argument("--directory", default="data/generated/league-v1")
    parser.add_argument("--reports", default="docs/training/league-v1")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--stages", type=int, default=4)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--games", type=int, default=4096)
    parser.add_argument("--later-games", type=int, default=8192)
    parser.add_argument("--snapshot-pool", type=int, default=4)
    parser.add_argument("--slots", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.00003)
    parser.add_argument("--selfplay-fraction", type=float, default=0.5)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--anchor-coef", type=float, default=0.02)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=310)
    parser.add_argument("--train-start", type=int, default=10000000)
    parser.add_argument("--dev-start", type=int, default=20000000)
    parser.add_argument("--screen-start", type=int, default=30000000)
    parser.add_argument("--final-start", type=int, default=40000000)
    parser.add_argument("--seed-stride", type=int, default=1000000)
    parser.add_argument("--dev-seeds", type=int, default=40)
    parser.add_argument("--screen-seeds", type=int, default=400)
    parser.add_argument("--final-seeds", type=int, default=1000)
    parser.add_argument("--regression-tolerance", type=float, default=3.0)
    parser.add_argument("--commit-reports", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.sampling_temperature) or args.sampling_temperature <= 0:
        parser.error("sampling temperature must be finite and positive")
    if not 0 <= args.selfplay_fraction <= 1:
        parser.error("selfplay fraction must be in [0,1]")
    if any(
        not math.isfinite(c) or c < 0
        for c in (args.value_coef, args.entropy_coef, args.anchor_coef)
    ):
        parser.error("PPO loss coefficients must be finite and nonnegative")
    if (
        min(
            args.stages,
            args.generations,
            args.games,
            args.later_games,
            args.slots,
            args.dev_seeds,
            args.screen_seeds,
            args.final_seeds,
            args.seed_stride,
        )
        < 1
    ):
        parser.error("counts must be positive")
    if args.snapshot_pool < 0 or args.lr <= 0 or args.regression_tolerance < 0:
        parser.error("invalid pool, learning rate or tolerance")
    ranges = [
        (
            args.train_start + i * args.seed_stride,
            args.train_start
            + i * args.seed_stride
            + args.generations * (args.games if i == 0 else args.later_games),
        )
        for i in range(args.stages)
    ]
    ranges += [
        (start + i * args.seed_stride, start + i * args.seed_stride + count)
        for start, count in (
            (args.dev_start, args.dev_seeds),
            (args.screen_start, args.screen_seeds),
        )
        for i in range(args.stages)
    ]
    ranges.append((args.final_start, args.final_start + args.final_seeds))
    ordered = sorted(ranges)
    if any(a[1] > b[0] for a, b in pairwise(ordered)):
        parser.error("all training/development/screen/final seed ranges must be disjoint")
    try:
        league(args)
    except Exception as exc:
        root = Path(args.directory)
        root.mkdir(parents=True, exist_ok=True)
        atomic_json(root / "league-error.json", {"error": str(exc), "type": type(exc).__name__})
        progress_path = root / "league-progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        atomic_json(progress_path, {**progress, "status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
