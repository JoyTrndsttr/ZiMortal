"""Two-decision cash teacher using independent, replay-consistent public particles.

PASS can reach different public information sets. Such sets are never pooled
as a clairvoyant second action; root max-Q remains unresolved in that case.
"""

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from statistics import NormalDist

import numpy as np
import torch

from zimortal.belief.conditional import HistorySampler
from zimortal.belief.sampling import SamplingFailure
from zimortal.engine import RuleClarificationRequired, RuleEngine
from zimortal.engine.observation import Observation
from zimortal.engine.tiles import tile_name

from .active import atomic_json, serialize_observation
from .cashq import digest
from .rollout_engine import RolloutEngine
from .runtime import choose_batch, load_model


def action_data(action):
    return json.loads(json.dumps(asdict(action))) if action is not None else None


def label(action):
    if action is None:
        return "terminal"
    text = action.kind.value
    if action.tile is not None:
        text += " " + tile_name(action.tile)
    if action.chi:
        text += " " + "".join(tile_name(t) for t in action.chi)
    if action.bi:
        text += " bi " + "/".join("".join(map(tile_name, g)) for g in action.bi)
    return text


def visible_key(obs):
    encoded = json.dumps(serialize_observation(obs), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def paired_statistics(outcomes, *, alpha=0.05, looks=1):
    """Common-particle normal intervals, Bonferroni over pairs and planned looks.

    Approximate MC diagnostics, not distribution-free sequential certification.
    No observed tie or zero variance is turned into a certified ordering.
    """
    values = np.asarray(outcomes, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("finite particle-by-branch outcomes, at least two particles required")
    n, k = values.shape
    means = values.mean(0)
    best = int(means.argmax())
    comparisons = max(1, k * (k - 1) // 2)
    z = NormalDist().inv_cdf(1 - alpha / (2 * comparisons * looks))
    rows = []
    resolved = k == 1
    for i in range(k):
        se = float(values[:, i].std(ddof=1) / math.sqrt(n))
        differences = values[:, i] - values[:, best]
        paired_se = float(differences.std(ddof=1) / math.sqrt(n))
        mean = float(differences.mean())
        interval = [mean - z * paired_se, mean + z * paired_se]
        rows.append(
            {
                "mean_cash": float(means[i]),
                "standard_error": se,
                "paired_vs_best": mean,
                "paired_standard_error": paired_se,
                "paired_ci": interval,
            }
        )
    if k > 1:
        resolved = all(
            i == best or (r["paired_standard_error"] > 0 and r["paired_ci"][1] < 0)
            for i, r in enumerate(rows)
        )
    return {
        "rows": rows,
        "best_index": best,
        "resolved": resolved,
        "rollouts": n,
        "interval_method": "approximate normal; Bonferroni over pairs and planned looks",
        "alpha": alpha,
        "status": "resolved" if resolved else "insufficient evidence",
    }


def advance_to_decision(
    engine,
    states,
    player,
    model,
    *,
    device="cpu",
    stop_at_player=True,
    timings=None,
    batch_size=128,
    limit=1000,
):
    """Immutable transitions, sole legal actions execute before the next free choice."""
    states = list(states)
    timings = timings if timings is not None else {}
    for _ in range(limit):
        waiting, observations = [], []
        progressed = False
        for i, state in enumerate(states):
            if state.terminal:
                continue
            actions = engine.legal_actions(state)
            if stop_at_player and len(actions) > 1 and actions[0].player == player:
                continue
            progressed = True
            if len(actions) == 1:
                states[i] = engine.step(state, actions[0])
            else:
                waiting.append(i)
                observations.append(engine.observation(state, actions[0].player))
        for start in range(0, len(waiting), batch_size):
            began = time.perf_counter()
            chosen = choose_batch(model, observations[start : start + batch_size], device=device)
            timings["neural_inference_seconds"] = (
                timings.get("neural_inference_seconds", 0) + time.perf_counter() - began
            )
            for i, action in zip(waiting[start : start + batch_size], chosen, strict=True):
                states[i] = engine.step(states[i], action)
        if not progressed:
            return states
    raise RuntimeError("two-step continuation exceeded transition limit")


def _summary(raw, *, alpha, looks):
    groups = {}
    for particle in raw:
        for branch in particle["branches"]:
            key = (branch["first_index"], branch["information_set"])
            group = groups.setdefault(
                key,
                {
                    "first_action": branch["first_action"],
                    "first_index": branch["first_index"],
                    "information_set": branch["information_set"],
                    "observation": branch["observation"],
                    "samples": {},
                },
            )
            group["samples"].setdefault(particle["particle"], []).append(branch)
    summaries = []
    # Allocate the root error budget over all encountered information sets, too.
    for group in groups.values():
        samples = group.pop("samples")
        ordered = [samples[p] for p in sorted(samples)]
        actions = [r["second_action"] for r in ordered[0]]
        if any([r["second_action"] for r in rows] != actions for rows in ordered):
            raise RuntimeError("one public information set produced inconsistent legal actions")
        if len(ordered) < 2:
            summaries.append(
                dict(
                    **group,
                    second_actions=[],
                    rollouts=len(ordered),
                    resolved=False,
                    status="insufficient evidence",
                )
            )
            continue
        outcomes = [[b["cash"] for b in rows] for rows in ordered]
        stats = paired_statistics(outcomes, alpha=alpha / max(1, len(groups)), looks=looks)
        second = [dict(action=a, **s) for a, s in zip(actions, stats["rows"], strict=True)]
        summaries.append(
            dict(
                **group,
                second_actions=second,
                rollouts=len(ordered),
                resolved=stats["resolved"],
                status=stats["status"],
                interval_method=stats["interval_method"],
                best_observed_action=actions[stats["best_index"]],
                best_supported_action=actions[stats["best_index"]] if stats["resolved"] else None,
            )
        )
    stable = [s for s in summaries if s["rollouts"] == len(raw)]
    root_candidates = []
    for s in stable:
        for a in s["second_actions"]:
            root_candidates.append((s["first_index"], s["information_set"], a["action"]))
    # Global max only if every first action has one identical visible second state.
    all_stable = len(stable) == len(groups) and len({s["first_index"] for s in stable}) == len(
        stable
    )
    root_stats = None
    if all_stable and root_candidates:
        outcomes = []
        for particle in raw:
            outcomes.append(
                [
                    next(
                        b["cash"]
                        for b in particle["branches"]
                        if b["first_index"] == first
                        and b["information_set"] == info
                        and b["second_action"] == action
                    )
                    for first, info, action in root_candidates
                ]
            )
        root_stats = paired_statistics(outcomes, alpha=alpha, looks=looks)
        root_stats["sequences"] = [
            {"first_index": f, "information_set": i, "second_action": a}
            for f, i, a in root_candidates
        ]
    return {
        "groups": summaries,
        "root_statistics": root_stats,
        "resolved": bool(root_stats and root_stats["resolved"]),
        "status": "resolved" if root_stats and root_stats["resolved"] else "insufficient evidence",
        "dynamic_information_sets": not all_stable,
    }


def evaluate_two_step_branches(
    obs,
    *,
    model,
    rollouts=128,
    seed=0,
    device="cpu",
    minimum=16,
    alpha=0.05,
    max_attempts=2000,
    chunk_size=8,
    sampler=None,
    engine=None,
    adaptive=True,
):
    if not isinstance(obs, Observation):
        raise TypeError("branch teacher accepts Observation only")
    if model is None or not obs.legal_actions or rollouts < 2 or minimum < 2:
        raise ValueError("frozen model, legal actions and at least two particles required")
    if not 0 < alpha < 1 or chunk_size < 1:
        raise ValueError("invalid alpha/chunk size")
    engine = engine or RolloutEngine()
    sampler = sampler or HistorySampler(obs)
    schedule = []
    n = min(minimum, rollouts)
    while n < rollouts:
        schedule.append(n)
        n *= 2
    schedule.append(rollouts)
    began = time.perf_counter()
    timing = {"sampling_seconds": 0.0, "neural_inference_seconds": 0.0}
    raw, terminal_count, attempts = [], 0, 0
    result = None
    current_particle = 0
    try:
        for target in schedule:
            while len(raw) < target:
                roots = []
                for _ in range(min(chunk_size, target - len(raw))):
                    current_particle = len(raw) + len(roots)
                    t = time.perf_counter()
                    world, used = sampler.sample(seed + current_particle * 104729, max_attempts)
                    timing["sampling_seconds"] += time.perf_counter() - t
                    attempts += used
                    world.validate()
                    if engine.observation(world, obs.player) != obs:
                        raise ValueError("particle does not match original public observation")
                    roots.append(world)
                first_states, descriptors = [], []
                for p, world in enumerate(roots):
                    before = world.serialize()
                    for first_index, action in enumerate(obs.legal_actions):
                        first_states.append(engine.step(world, action))
                        descriptors.append((p, first_index))
                    if world.serialize() != before:
                        raise RuntimeError("branch expansion mutated shared particle")
                first_states = advance_to_decision(
                    engine, first_states, obs.player, model, device=device, timings=timing
                )
                terminals, tags = [], []
                partial = [
                    {
                        "particle": len(raw) + p,
                        "world_hash": hashlib.sha256(world.serialize().encode()).hexdigest(),
                        "branches": [],
                    }
                    for p, world in enumerate(roots)
                ]
                for state, (p, first_index) in zip(first_states, descriptors, strict=True):
                    first_action = action_data(obs.legal_actions[first_index])
                    if state.terminal:
                        branches = [(None, state)]
                        next_obs = None
                        key = "terminal"
                    else:
                        next_obs = engine.observation(state, obs.player)
                        if len(next_obs.legal_actions) < 2:
                            raise RuntimeError("did not stop at original player's free decision")
                        key = visible_key(next_obs)
                        branches = [(a, engine.step(state, a)) for a in next_obs.legal_actions]
                    for second, child in branches:
                        terminals.append(child)
                        tags.append(
                            (
                                p,
                                {
                                    "first_index": first_index,
                                    "first_action": first_action,
                                    "information_set": key,
                                    "observation": serialize_observation(next_obs)
                                    if next_obs
                                    else None,
                                    "second_action": action_data(second),
                                },
                            )
                        )
                terminals = advance_to_decision(
                    engine,
                    terminals,
                    obs.player,
                    model,
                    device=device,
                    stop_at_player=False,
                    timings=timing,
                )
                for state, (p, entry) in zip(terminals, tags, strict=True):
                    state.validate()
                    if not state.terminal:
                        raise RuntimeError("unfinished terminal branch")
                    entry["cash"] = state.settlement.payments[obs.player] if state.settlement else 0
                    entry["fan"] = state.settlement.fan if state.settlement else {}
                    entry["winner"] = state.settlement.winner if state.settlement else None
                    partial[p]["branches"].append(entry)
                terminal_count += len(terminals)
                raw.extend(partial)
            result = _summary(raw, alpha=alpha, looks=len(schedule))
            if adaptive and result["resolved"]:
                break
    except (SamplingFailure, RuleClarificationRequired, ValueError, RuntimeError) as exc:
        return {
            "valid": False,
            "resolved": False,
            "status": "failed root; no label",
            "error": str(exc),
            "error_type": type(exc).__name__,
            "failed_particle": current_particle,
            "completed_particles": len(raw),
            "raw_particles": raw,
            "sampling_attempts": attempts,
            "sampling_failures": int(isinstance(exc, SamplingFailure)),
            "rule_failures": int(isinstance(exc, RuleClarificationRequired)),
        }
    elapsed = time.perf_counter() - began
    return dict(
        **result,
        valid=True,
        rollouts=len(raw),
        sampling_attempts=attempts,
        sampling_failures=0,
        rule_failures=0,
        raw_particles=raw,
        first_actions=[action_data(a) for a in obs.legal_actions],
        timings=dict(
            **timing,
            total_seconds=elapsed,
            simulation_other_seconds=elapsed - sum(timing.values()),
            particles_per_second=len(raw) / elapsed,
            terminal_rollouts=terminal_count,
            terminal_rollouts_per_second=terminal_count / elapsed,
        ),
        continuation="frozen greedy policy for all seats; no stochastic action stream",
        units="actual terminal payments, no division or shaping",
        posterior="HistorySampler combinatorial prior conditioned on legal full public replay; not learned opponent likelihood",
    )


def diagnostic_observation(case_id, *, step=None):
    """Construct only the visible root; complete seed world never reaches teacher."""
    from zimortal.web.server import replay_frame

    feedback = json.loads(Path("docs/strategy/reviews/human-review-v1.json").read_text())
    record = next(r for r in feedback["records"] if r["case_id"] == case_id)
    prefix = record["history_prefix"]
    target = step or record["step"]
    if not 1 <= target <= record["step"]:
        raise ValueError("diagnostic step outside saved prefix")
    engine = RuleEngine()
    state = engine.new_game(record["seed"], record["dealer"])
    for action in prefix[: target - 1]:
        state = replay_frame(engine, state, {"action": action})
    state.validate()
    obs = engine.observation(state, record["player"])
    if target == record["step"] and list(obs.hand) != record["hand"]:
        raise ValueError("manual root hand mismatch")
    return obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/generated/league-v4/stage-02/best-dev.pt")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["split-001", "split-002", "split-005", "split-002-step-02", "split-002-step-13"],
    )
    parser.add_argument("--rollouts", type=int, default=128)
    parser.add_argument("--minimum", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=712)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--directory", default="data/generated/branch-teacher-v1")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    # Pure bounded caches only, no second sampler or changed rule semantics.
    from zimortal.engine import game

    from .rollout_engine import cached_chi, cached_evaluate_hand

    game.evaluate_hand = cached_evaluate_hand
    game.enumerate_chi_with_required_bi = cached_chi
    from .online_rl import source_hashes

    root = Path(args.directory)
    root.mkdir(parents=True, exist_ok=True)
    config = dict(**vars(args), sources=source_hashes(), model_sha256=digest(args.model))
    config["manual_case_sha256"] = digest("docs/strategy/reviews/human-review-v1.json")
    path = root / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("frozen source/config/model differs; use a new directory")
    atomic_json(path, config)
    model = load_model(args.model, args.device)
    for i, name in enumerate(args.cases):
        if (
            source_hashes() != config["sources"]
            or digest(args.model) != config["model_sha256"]
            or digest("docs/strategy/reviews/human-review-v1.json") != config["manual_case_sha256"]
        ):
            raise ValueError("frozen producer/model/manual roots changed during diagnostics")
        if (root / f"{name}.json").exists():
            continue
        case_id, _, step = name.partition("-step-")
        obs = diagnostic_observation(case_id, step=int(step) if step else None)
        profiler = None
        if args.profile:
            import cProfile

            profiler = cProfile.Profile()
            profiler.enable()
        atomic_json(
            root / "progress.json",
            {"status": "running", "case": name, "rollouts": args.rollouts, "device": args.device},
        )
        print(f"starting {name}: {len(obs.legal_actions)} first actions", flush=True)
        result = evaluate_two_step_branches(
            obs,
            model=model,
            rollouts=args.rollouts,
            minimum=args.minimum,
            seed=args.seed + i * 1000000,
            device=args.device,
            chunk_size=args.chunk_size,
            max_attempts=args.max_attempts,
            alpha=0.05 / len(args.cases),
        )
        if profiler:
            profiler.disable()
            profiler.dump_stats(str(root / f"{name}.profile"))
        raw = result.pop("raw_particles")
        atomic_json(root / f"{name}-particles.json", raw)
        atomic_json(
            root / f"{name}.json",
            dict(
                case_id=case_id,
                diagnostic_step=step or None,
                observation_sha256=visible_key(obs),
                model=args.model,
                model_sha256=config["model_sha256"],
                **result,
            ),
        )
        print(
            json.dumps(
                {
                    "case": name,
                    "valid": result["valid"],
                    "status": result["status"],
                    "timings": result.get("timings"),
                    "error": result.get("error"),
                }
            ),
            flush=True,
        )
        if not result["valid"]:
            atomic_json(
                root / "progress.json", {"status": "failed", "case": name, "error": result["error"]}
            )
            raise RuntimeError(f"failed diagnostic root: {name}: {result['error']}")
    atomic_json(
        root / "progress.json",
        {"status": "complete", "cases": args.cases, "training_started": False},
    )


if __name__ == "__main__":
    main()
