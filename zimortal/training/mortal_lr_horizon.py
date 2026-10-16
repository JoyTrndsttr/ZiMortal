"""Single-factor LR timescale comparison; no model/behavior/target changes."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .active import atomic_json
from .mortal_boltzmann import AnchoredBoltzmann
from .mortal_boltzmann_experiment import behavior_summary
from .mortal_continuation_experiment import evaluate
from .mortal_data import audit_dataset, load_rows, save_shard
from .mortal_deviation_stats import cluster_cohort
from .mortal_diagnostics import distribution
from .mortal_extraction_experiment import CHECKPOINT
from .mortal_offline import BASELINE, regression
from .mortal_online import checkpoint, collect, learning_rate, make_optimizer, update
from .mortal_online_long import evidence_summary, parameter_change, q_probe
from .mortal_sarsa import parameter_hash
from .online_rl import digest, source_hashes
from .runtime import load_model

GENERATIONS, GAMES, BATCH = 10, 5000, 256
HISTORY = Path("data/generated/mortal-online-long-v1")


def horizon_protocol(history=HISTORY):
    """Lock a forecast from COMPLETED historical row counts, never new outcomes."""
    files = [history / f"gen-{g:02d}/training-result.json" for g in range(1, 11)]
    historical_rows = sum(json.loads(p.read_text())["decisions"] for p in files)
    horizon = GENERATIONS * math.ceil(GAMES * historical_rows / 50000 / BATCH)
    return {
        "historical_rows": historical_rows,
        "historical_games": 50000,
        "rows_per_game": historical_rows / 50000,
        "expected_updates": horizon,
        "warmup_updates": 20,
        "A_cosine_updates": 300,
        "B_cosine_updates": horizon - 21,
        "B_first_floor_step_zero_based": horizon - 1,
        "history_sha256": {str(p): digest(p) for p in files},
        "actual_count_rule": "no truncation/replay/rescaling; clamp at floor if actual steps exceed forecast; report shortfall if fewer",
    }


def schedule(step, cosine_updates):
    if step < 0 or cosine_updates < 1:
        raise ValueError("nonnegative step and positive cosine length required")
    if step < 20:
        return learning_rate(step)
    phase = min(1.0, (step - 20) / cosine_updates)
    return 3e-6 + 0.5 * (3e-5 - 3e-6) * (1 + math.cos(math.pi * phase))


def seed_specs(generation=None, *, evaluation=False, holdout=False):
    if holdout:
        if generation is not None or evaluation:
            raise ValueError("holdout uses its own locked block")
        start, count = 1290000000, 5000
    else:
        if generation not in range(1, 11):
            raise ValueError("generation must be 1..10")
        start = (1280000000 if evaluation else 1270000000) + generation * 100000
        count = 1000 if evaluation else GAMES
    if evaluation or holdout:
        return [(s, s % 3, seat, 0) for s in range(start, start + count) for seat in range(3)]
    return [(start + i, i % 3, (i // 3) % 3, 0) for i in range(count)]


def seed_lock():
    lock = {
        "generations": {
            str(g): {"train_A_B": seed_specs(g), "evaluation": seed_specs(g, evaluation=True)}
            for g in range(1, 11)
        },
        "holdout": seed_specs(holdout=True),
    }
    return lock


def prior_seed_check(lock):
    def integers(value):
        if isinstance(value, dict):
            for v in value.values():
                yield from integers(v)
        elif isinstance(value, list):
            for v in value:
                yield from integers(v)
        elif isinstance(value, int):
            yield value

    seeds = {v for v in integers(lock) if v >= 1270000000}
    checked = {}
    for p in sorted(Path("data/generated").glob("*/seed-lock.json")):
        if seeds.intersection(integers(json.loads(p.read_text()))):
            raise RuntimeError(f"seed overlap with {p}")
        checked[str(p)] = digest(p)
    return checked


def buffer_snapshot(model):
    return {
        n: v.detach().cpu().clone()
        for n, v in model.named_buffers()
        if n.endswith(("running_mean", "running_var", "num_batches_tracked"))
    }


def bn_change(before, model):
    after = dict(model.named_buffers())
    result = {}
    for suffix in ("running_mean", "running_var", "num_batches_tracked"):
        keys = [n for n in before if n.endswith(suffix)]
        delta = [(after[n].detach().cpu().double() - before[n].double()).flatten() for n in keys]
        base = [before[n].double().flatten() for n in keys]
        result[suffix] = {
            "buffers": len(keys),
            "delta_l2": float(torch.cat(delta).norm()) if delta else 0.0,
            "previous_l2": float(torch.cat(base).norm()) if base else 0.0,
        }
    return result


def q_dynamics(before, after, rows):
    """Legal-action-weighted orthogonal decomposition; common shift is NOT true Vπ."""
    if not rows or not len(before) == len(after) == len(rows):
        raise ValueError("nonempty identical probe rows required")
    delta = [
        np.asarray(b, float) - np.asarray(a, float) for a, b in zip(before, after, strict=True)
    ]
    common = sum(len(d) * float(d.mean()) ** 2 for d in delta)
    advantage = sum(float(((d - d.mean()) ** 2).sum()) for d in delta)
    count = sum(map(len, delta))

    def anchor(q, r):
        p = np.asarray(r["reference_probabilities"], np.float32)
        return int(np.argmax(q + np.float32(0.3) * np.log(p + np.float32(1e-8))))

    total = common + advantage
    return {
        "inputs": len(rows),
        "legal_candidates": count,
        "legal_q_rmse": math.sqrt(total / count),
        "common_shift_rms": math.sqrt(common / count),
        "centered_advantage_rms": math.sqrt(advantage / count),
        "common_square_fraction": common / total if total else 0.0,
        "advantage_square_fraction": advantage / total if total else 0.0,
        "raw_argmax_change_rate": float(
            np.mean([a.argmax() != b.argmax() for a, b in zip(before, after, strict=True)])
        ),
        "anchored_argmax_change_rate": float(
            np.mean(
                [anchor(a, r) != anchor(b, r) for a, b, r in zip(before, after, rows, strict=True)]
            )
        ),
        "scope": "same current A-generation visible probe inputs for both arms; no probe loss/replay; not generalization",
    }


def optimizer_dynamics(losses, optimizer):
    lrs = [l["lr"] for l in losses]
    steps = sorted({int(v["step"]) for v in optimizer.state.values()})
    return {
        "updates_this_generation": len(losses),
        "cumulative_updates": losses[-1]["update"],
        "last_actual_lr": lrs[-1],
        "mean_lr": float(np.mean(lrs)),
        "min_lr": min(lrs),
        "max_lr": max(lrs),
        # Warmup starts at the same floor; separately count post-warmup floor steps.
        "floor_updates": sum(v <= 3e-6 + 1e-18 for v in lrs),
        "post_warmup_floor_updates": sum(
            l["update"] - 1 >= 20 and l["lr"] <= 3e-6 + 1e-18 for l in losses
        ),
        "gradient_norm": distribution([l["gradient_norm"] for l in losses]),
        "clip_fraction": float(np.mean([l["gradient_norm"] > 1 for l in losses])),
        "adam_steps": steps,
    }


def extreme_stop(results):
    return any(
        len(results) >= 2
        and all(r["cash"][f"{arm}-anchor"]["seed_bootstrap_95"][1] < -5 for r in results[-2:])
        for arm in ("A", "B")
    )


def evaluate_and_diagnose(
    root, specs, models, initial, parent, reference, frozen, parent_hash, *, holdout=False
):
    result = evaluate(
        root, specs, models, initial, parent, reference, frozen, parent_hash, holdout=holdout
    )
    outcomes = json.loads((root / "screening-outcomes.json").read_text())
    if not holdout:
        for arm in models:
            anchor = {(g["seed"], g["seat"]): g for g in outcomes[f"{arm}-anchor"]}
            result["contrasts"][f"{arm}_raw_minus_anchor"] = cluster_cohort(
                outcomes[f"{arm}-raw"], anchor
            )
    atomic_json(root / "result.json", result)
    return result


def run(directory):
    torch.set_num_threads(2)
    torch.manual_seed(20261015)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    root = Path(directory)
    if root.exists():
        raise RuntimeError("fresh experiment directory required")
    protocol, lock = horizon_protocol(), seed_lock()
    prior = prior_seed_check(lock)
    old = json.loads(Path("data/generated/mortal-online-v1/config.json").read_text())
    for p, sha in old["frozen_sha256"].items():
        if (
            p.startswith(("zimortal/engine/", "zimortal/model/")) or p in (BASELINE, CHECKPOINT)
        ) and digest(p) != sha:
            raise RuntimeError("fixed rules/model/start changed")
    frozen = source_hashes() | {p: digest(p) for p in (BASELINE, CHECKPOINT)}
    root.mkdir(parents=True)
    config = {
        "experiment": "single-factor online LR horizon v1",
        "generations": 10,
        "games_per_arm_generation": 5000,
        "schedule": protocol,
        "reference": old["behavior"],
        "optimizer_loss_protocol": old,
        "only_variable": "cosine decay length A=300 vs B=prelocked forecast-21; step zero-based",
        "formula": "warmup s<20:3e-6+27e-6*s/20; else 3e-6+13.5e-6*(1+cos(pi*min(1,(s-20)/L)))",
        "actual_steps": "ceil(actual real rows/256) per generation; all rows exactly once; optimizer and RNG continuous",
        "temperature": 0.5,
        "lambda": 0.3,
        "loss": "terminal cash/100 half MSE only",
        "CQL": 0,
        "replay": False,
        "HU_PASS": "same conservative anchored greedy",
        "BN": "unchanged training running statistics",
        "probe": "4096 evenly spaced rows in A current generation order; same visible inputs for both arms, before/after; labels unused",
        "evaluation": "1000 new seeds x3 seats x6 policies per generation; 10000 seed-cluster bootstrap",
        "final_candidate": "A-Gen10 and B-Gen10 only; no intermediate selection",
        "final_holdout": "5000 locked new seeds x3 seats; baseline/initial/A10/B10 once only",
        "primary": "final B-A and B-initial lower95>0 supports important LR bottleneck; also report B-baseline",
        "safety": "either arm anchor-baseline CI upper<-5 for two consecutive generations; stop both, holdout unused",
        "no_automatic_promotion": True,
        "frozen_sha256": frozen,
        "prior_seed_locks": prior,
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(0),
    }
    atomic_json(root / "config.json", config)
    atomic_json(root / "seed-lock.json", lock)
    parent = load_model(BASELINE, "cuda").requires_grad_(False)
    initial = load_model(CHECKPOINT, "cuda").requires_grad_(False)
    models = {a: load_model(CHECKPOINT, "cuda") for a in ("A", "B")}
    if len({parameter_hash(m) for m in (*models.values(), initial)}) != 1:
        raise RuntimeError("different initial models")
    optimizers = {a: make_optimizer(m) for a, m in models.items()}
    rngs = {a: np.random.default_rng(20261015) for a in models}
    torch.manual_seed(20261015)
    torch_rngs = {
        a: (torch.get_rng_state().clone(), [s.clone() for s in torch.cuda.get_rng_state_all()])
        for a in models
    }
    updates, floor_counts, post_floor_counts = (dict.fromkeys(models, 0) for _ in range(3))
    parent_hash, results = parameter_hash(parent), []
    for generation in range(1, 11):
        gen = root / f"gen-{generation:02d}"
        gen.mkdir()
        manifests, producers, audits, behaviors, entropies, execution_cash = {}, {}, {}, {}, {}, {}
        for arm, model in models.items():
            armroot = gen / arm
            armroot.mkdir()
            data_root = armroot / "data"
            data_root.mkdir()
            actor = AnchoredBoltzmann(model, parent, config["reference"])
            producers[arm] = parameter_hash(model)
            manifest = {
                "shards": [],
                "arm": arm,
                "generation": generation,
                "actor_state_sha256": producers[arm],
                "producer_checkpoint": CHECKPOINT
                if generation == 1
                else str(root / f"gen-{generation - 1:02d}" / arm / "checkpoint.pt"),
            }
            records, games = [], []
            specs = seed_specs(generation)
            for start in range(0, GAMES, 300):
                atomic_json(
                    root / "progress.json",
                    {
                        "phase": "collection",
                        "generation": generation,
                        "arm": arm,
                        "completed": start,
                        "total": GAMES,
                    },
                )
                batch_games, batch_records, data = collect(
                    actor,
                    parent,
                    specs[start : start + 300],
                    armroot / f"evidence-{start // 300:02d}",
                    f"{arm}-generation-{generation}",
                    training_rows=True,
                    device="cuda",
                )
                manifest["shards"].append(save_shard(data_root, start // 300, data))
                games.extend(batch_games)
                records.extend(batch_records)
                atomic_json(data_root / "manifest.json", manifest)
            if parameter_hash(model) != producers[arm]:
                raise RuntimeError("collection changed producer/BN")
            atomic_json(armroot / "training-outcomes.json", games)
            atomic_json(
                root / "progress.json",
                {"phase": "training_audit", "generation": generation, "arm": arm},
            )
            audits[arm] = audit_dataset(data_root, manifest)
            manifests[arm] = manifest
            behaviors[arm] = behavior_summary(records)
            entropies[arm] = evidence_summary(armroot)
            execution_cash[arm] = distribution([g["cash"] for g in games])
            del actor, records, games, batch_records, data
        all_rows = {a: load_rows(gen / a / "data", manifests[a]) for a in models}
        indices = np.linspace(
            0, len(all_rows["A"]) - 1, min(4096, len(all_rows["A"])), dtype=int
        ).tolist()
        probe = [all_rows["A"][i] for i in indices]
        atomic_json(
            gen / "probe-lock.json",
            {
                "A_row_indices": indices,
                "manifest_sha256": digest(gen / "A/data/manifest.json"),
                "same_inputs_for_both_arms": True,
                "no_probe_loss": True,
            },
        )
        arms = {}
        for arm, model in models.items():
            rows = all_rows[arm]
            before = regression(model, rows, device="cuda")
            q_before = q_probe(model, probe)
            parameters = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
            buffers = buffer_snapshot(model)
            atomic_json(
                root / "progress.json",
                {
                    "phase": "MC_update",
                    "generation": generation,
                    "arm": arm,
                    "previous_updates": updates[arm],
                },
            )
            cpu_rng, cuda_rng = torch_rngs[arm]
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state_all(cuda_rng)
            length = 300 if arm == "A" else protocol["B_cosine_updates"]
            updates[arm], losses = update(
                model,
                optimizers[arm],
                rows,
                rngs[arm],
                updates[arm],
                device="cuda",
                lr_schedule=lambda step, length=length: schedule(step, length),
            )
            torch_rngs[arm] = (
                torch.get_rng_state().clone(),
                [s.clone() for s in torch.cuda.get_rng_state_all()],
            )
            dynamics = optimizer_dynamics(losses, optimizers[arm])
            floor_counts[arm] += dynamics["floor_updates"]
            post_floor_counts[arm] += dynamics["post_warmup_floor_updates"]
            dynamics.update(
                cumulative_floor_updates=floor_counts[arm],
                cumulative_post_warmup_floor_updates=post_floor_counts[arm],
            )
            cp = gen / arm / "checkpoint.pt"
            checkpoint(
                model,
                optimizers[arm],
                cp,
                updates[arm],
                rngs[arm],
                arm=arm,
                generation=generation,
                producer_state_sha256=producers[arm],
                manifest_sha256=digest(gen / arm / "data/manifest.json"),
                source_sha256=frozen,
                seed_lock_sha256=digest(root / "seed-lock.json"),
                learning_rate_schedule={
                    "warmup": 20,
                    "cosine_updates": length,
                    "forecast_updates": protocol["expected_updates"],
                },
            )
            atomic_json(gen / arm / "updates.json", losses)
            arms[arm] = {
                "checkpoint_sha256": digest(cp),
                "optimizer_sha256": digest(gen / arm / "optimizer.pt"),
                "producer_state_sha256": producers[arm],
                "updated_state_sha256": parameter_hash(model),
                "training_audit": audits[arm],
                "decisions": len(rows),
                "updates": updates[arm],
                "optimizer_dynamics": dynamics,
                "parameter_change": parameter_change(parameters, model),
                "bn_change": bn_change(buffers, model),
                "q_drift": q_dynamics(q_before, q_probe(model, probe), probe),
                "regression_before": before,
                "regression_after": regression(model, rows, device="cuda"),
                "behavior": behaviors[arm],
                "diagnostics": entropies[arm],
                "terminal_cash_yuan": execution_cash[arm],
            }
            atomic_json(gen / arm / "training-result.json", arms[arm])
            del parameters, buffers, q_before, losses
        del all_rows, probe, rows
        atomic_json(root / "progress.json", {"phase": "evaluation", "generation": generation})
        evaluation = evaluate_and_diagnose(
            gen / "evaluation",
            seed_specs(generation, evaluation=True),
            models,
            initial,
            parent,
            config["reference"],
            frozen,
            parent_hash,
        )
        summary = {"generation": generation, "arms": arms, **evaluation}
        results.append(summary)
        atomic_json(gen / "result.json", summary)
        atomic_json(root / "generations.json", results)
        print(
            json.dumps(
                {
                    "generation": generation,
                    "cash": summary["cash"],
                    "contrasts": summary["contrasts"],
                }
            ),
            flush=True,
        )
        if extreme_stop(results):
            break
    completed = len(results) == 10
    final = None
    if completed:
        atomic_json(
            root / "final-candidate-lock.json",
            {
                "generation": 10,
                **{f"{a}_sha256": digest(root / f"gen-10/{a}/checkpoint.pt") for a in models},
                "seed_lock_sha256": digest(root / "seed-lock.json"),
            },
        )
        atomic_json(root / "progress.json", {"phase": "final_holdout", "generation": 10})
        final = evaluate_and_diagnose(
            root / "final-holdout",
            seed_specs(holdout=True),
            models,
            initial,
            parent,
            config["reference"],
            frozen,
            parent_hash,
            holdout=True,
        )
    supports = bool(
        completed
        and final["contrasts"]["B_minus_A"]["seed_bootstrap_95"][0] > 0
        and final["contrasts"]["B_minus_initial"]["seed_bootstrap_95"][0] > 0
    )
    atomic_json(
        root / "active-checkpoint.json",
        {"checkpoint": CHECKPOINT, "sha256": digest(CHECKPOINT), "baseline_replaced": False},
    )
    atomic_json(
        root / "result.json",
        {
            "status": "complete_stop" if completed else "extreme_safety_stop_rollback",
            "generations": results,
            "final_holdout": final,
            "supports_important_LR_bottleneck": supports,
            "baseline_replaced": False,
            "automatic_expansion": False,
        },
    )
    atomic_json(
        root / "progress.json",
        {
            "phase": "complete_stop" if completed else "extreme_safety_stop_rollback",
            "generations": len(results),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="data/generated/mortal-lr-horizon-v1")
    run(parser.parse_args().directory)
