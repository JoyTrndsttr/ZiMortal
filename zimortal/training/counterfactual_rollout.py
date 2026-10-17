"""Frozen, fixed-hidden-hands ranking microscope. Never a training-data producer.

All root actions share each shuffled remaining wall. Models see Observation only.
One common offline-anchor continuation defines the same estimand for every critic.
"""

import hashlib
import json
import math
from itertools import combinations, pairwise

import numpy as np
import torch

from zimortal.engine import RuleEngine
from zimortal.model.encoding import encode_action, encode_observation
from zimortal.training.mortal_counterfactual import correlation, ranks
from zimortal.training.mortal_credit_diagnostics import blob, summary
from zimortal.training.mortal_extraction import ExtractionPolicy
from zimortal.training.mortal_sarsa import parameter_hash
from zimortal.training.online_rl import pack

SEED = 20261018
WORLDS = 32
BOOTSTRAP = 4096
QUOTAS = {"discard": 200, "chi/pass": 125, "peng/pass": 125, "hu/pass": 50}


def seeded(*parts):
    return int.from_bytes(hashlib.sha256(blob([SEED, *parts])).digest()[:8], "little")


def state_digest(state):
    return hashlib.sha256(blob(state.serialize())).hexdigest()


def pick_roots(pool):
    """Stratified random selection from an earlier outcome-blind uniform sample."""
    rng = np.random.default_rng(SEED)
    result = []
    for family, count in QUOTAS.items():
        rows = sorted((r for r in pool if r["family"] == family), key=lambda r: r["locator"])
        if len(rows) < count:
            raise ValueError(f"insufficient locked population for {family}")
        result.extend(rows[int(i)] for i in sorted(rng.choice(len(rows), count, replace=False)))
    return [result[int(i)] for i in rng.permutation(len(result))]


def world_record(root_id, world):
    return {
        "world_id": f"{root_id}:{world}",
        "hidden_state_seed": seeded(root_id, world, "hidden"),
        "wall_seed": seeded(root_id, world, "wall"),
        "continuation_rng_seed": seeded(root_id, world, "continuation"),
        "hidden_state_seed_usage": "unused: original opponent hidden hands fixed",
        "continuation_rng_usage": "unused: deterministic greedy policies",
    }


def make_world(root, root_id, world):
    out = root.clone()
    record = world_record(root_id, world)
    np.random.default_rng(record["wall_seed"]).shuffle(out.deck)
    out.validate()
    record.update(deck=out.deck.copy(), state_sha256=state_digest(out))
    return out, record


def visible_batch(model, observations, device):
    features = [encode_observation(o, model.feature_version) for o in observations]
    actions = [
        np.stack([encode_action(a, o.player) for a in o.legal_actions]) for o in observations
    ]
    return pack(features, actions, device)


def frozen_predictions(model, observations, device):
    with torch.inference_mode():
        scores = model(*visible_batch(model, observations, device))[0]
    return [
        a[: len(o.legal_actions)].cpu().numpy().copy()
        for a, o in zip(scores, observations, strict=True)
    ]


def root_predictions(root, models, baseline, behavior, *, device):
    engine = RuleEngine()
    obs = engine.observation(root, engine.legal_actions(root)[0].player)
    x, a, mask = visible_batch(baseline, [obs], device)
    from .mortal_support import behavior_probability

    with torch.inference_mode():
        logits = baseline(x, a, mask)[0]
        p = behavior_probability(logits, a, mask, behavior)
        result = {"baseline": logits[0].cpu().tolist()}
        for name, model in models.items():
            q = model.forward_q(x, a, mask)[0]
            result[name] = {
                "q": q[0].cpu().tolist(),
                "anchor": (q + 0.3 * torch.log(p + 1e-8))[0].cpu().tolist(),
            }
    return result, p[0].cpu().tolist(), int(logits.argmax(-1)[0])


def continue_root(root, root_id, anchor, baseline, *, engine=None, device="cuda"):
    """Exactly 32 worlds/action. Audit repeats these SAME worlds, not new samples."""
    engine = engine or RuleEngine()
    original = state_digest(root)
    legal = engine.legal_actions(root)
    player = legal[0].player
    states, worlds = [], []
    stats = {
        "future_steps": 0,
        "gpu_batch_calls": 0,
        "gpu_model_forward_calls": 0,
        "neural_observations": 0,
        "trainee_observations": 0,
        "opponent_observations": 0,
    }
    for j in range(WORLDS):
        world, record = make_world(root, root_id, j)
        if engine.observation(world, player) != engine.observation(root, player):
            raise RuntimeError("wall shuffle changed visible information")
        worlds.append(record)
        for action in legal:
            states.append(engine.step(world, action))
            stats["future_steps"] += 1
    for _ in range(1000):
        waiting = {"trainee": [], "opponent": []}
        for i, state in enumerate(states):
            if state.terminal:
                continue
            actions = engine.legal_actions(state)
            if len(actions) == 1:
                states[i] = engine.step(state, actions[0])
                stats["future_steps"] += 1
            else:
                role = "trainee" if actions[0].player == player else "opponent"
                waiting[role].append((i, engine.observation(state, actions[0].player)))
        for role, model in (("trainee", anchor), ("opponent", baseline)):
            for start in range(0, len(waiting[role]), 256):
                batch = waiting[role][start : start + 256]
                predictions = frozen_predictions(model, [o for _, o in batch], device)
                stats["gpu_batch_calls"] += 1
                stats["gpu_model_forward_calls"] += 2 if role == "trainee" else 1
                stats["neural_observations"] += len(batch)
                stats[f"{role}_observations"] += len(batch)
                for (i, obs), scores in zip(batch, predictions, strict=True):
                    states[i] = engine.step(states[i], obs.legal_actions[int(scores.argmax())])
                    stats["future_steps"] += 1
        if all(s.terminal for s in states):
            break
    else:
        raise RuntimeError("benchmark continuation exceeded 1000 deterministic ticks")
    evidence, outcomes = [], np.empty((WORLDS, len(legal)))
    prefix = len(root.history)
    for index, terminal in enumerate(states):
        terminal.validate()
        cash = terminal.settlement.payments[player] if terminal.settlement else 0
        j, action = divmod(index, len(legal))
        outcomes[j, action] = cash
        evidence.append(
            {
                "world": j,
                "action": action,
                "cash": cash,
                "terminal_sha256": state_digest(terminal),
                "suffix": json.loads(blob(terminal.history[prefix:])),
            }
        )
    assert state_digest(root) == original
    return outcomes, {"worlds": worlds, "branches": evidence, "counts": stats}


def ev_statistics(outcomes, seed):
    values = np.asarray(outcomes, float)
    if values.shape[0] != WORLDS or values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("complete fixed-size world by action array required")
    mean = values.mean(0)
    std = values.std(0, ddof=1)
    se = std / math.sqrt(WORLDS)
    indices = np.random.default_rng(seed).integers(WORLDS, size=(BOOTSTRAP, WORLDS))
    boot = values[indices].mean(1)
    intervals = np.quantile(boot, [0.025, 0.975], axis=0).T
    pairs = []
    for i, j in combinations(range(len(mean)), 2):
        delta = values[:, i] - values[:, j]
        ci = np.quantile(boot[:, i] - boot[:, j], [0.025, 0.975]).tolist()
        pairs.append(
            {
                "i": i,
                "j": j,
                "mean": float(delta.mean()),
                "std": float(delta.std(ddof=1)),
                "se": float(delta.std(ddof=1) / math.sqrt(WORLDS)),
                "ci95": ci,
            }
        )
    order = np.argsort(-mean, kind="stable")
    top, second = map(int, order[:2])
    delta = values[:, top] - values[:, second]
    ci = np.quantile(boot[:, top] - boot[:, second], [0.025, 0.975]).tolist()
    return {
        "actions": [
            {
                "mean": float(mean[i]),
                "std": float(std[i]),
                "se": float(se[i]),
                "ci95": intervals[i].tolist(),
                "median": float(np.median(values[:, i])),
                "p25": float(np.percentile(values[:, i], 25)),
                "p75": float(np.percentile(values[:, i], 75)),
            }
            for i in range(len(mean))
        ],
        "pairs": pairs,
        "top": top,
        "second": second,
        "top_second_delta": float(delta.mean()),
        "top_second_ci95": ci,
        "clear": ci[0] > 0,
        "best_set": np.flatnonzero(np.isclose(mean, mean.max(), atol=1e-10, rtol=0)).tolist(),
        "interval_method": "4096 paired-world percentile bootstrap; pointwise, no post-selection or multiplicity correction",
        "distinct_outcome_vectors": len(np.unique(values, axis=0)),
    }


def ranking(scores, ev, outcomes):
    scores = np.asarray(scores, float)
    values = np.array([a["mean"] for a in ev["actions"]])
    chosen = int(scores.argmax())
    order = np.argsort(-scores, kind="stable")
    i, j = map(int, order[:2])
    ps, vs, resolved = [], [], []
    for pair in ev["pairs"]:
        a, b = pair["i"], pair["j"]
        pred, actual = np.sign(scores[a] - scores[b]), np.sign(values[a] - values[b])
        if actual:
            ps.append(0.5 if pred == 0 else float(pred == actual))
            vs.append((pred, actual))
            if pair["ci95"][0] > 0 or pair["ci95"][1] < 0:
                resolved.append(ps[-1])
    all_pairs = [
        (np.sign(scores[a] - scores[b]), np.sign(values[a] - values[b]))
        for a, b in combinations(range(len(scores)), 2)
    ]
    numerator = sum(a * b for a, b in all_pairs)
    denominator = math.sqrt(sum(a != 0 for a, _ in all_pairs) * sum(b != 0 for _, b in all_pairs))
    pair = np.asarray(outcomes)[:, i] - np.asarray(outcomes)[:, j]
    return {
        "chosen": chosen,
        "top1_accuracy": float(chosen == ev["top"]),
        "best_action_hit_rate": float(chosen in ev["best_set"]),
        "pairwise_accuracy": float(np.mean(ps)) if ps else None,
        "pairwise_correct_sum": float(sum(ps)),
        "non_tied_pairs": len(ps),
        "resolved_pairwise_accuracy": float(np.mean(resolved)) if resolved else None,
        "resolved_pairs": len(resolved),
        "spearman": correlation(ranks(scores), ranks(values)),
        "kendall_tau_b": numerator / denominator if denominator else None,
        "regret_yuan": float(values.max() - values[chosen]),
        "chosen_EV_yuan": float(values[chosen]),
        "model_margin_native": float(scores[i] - scores[j]),
        "rollout_margin_yuan": float(pair.mean()),
        "rollout_margin_se_yuan": float(pair.std(ddof=1) / math.sqrt(WORLDS)),
    }


def mean_present(rows, key):
    v = [r[key] for r in rows if r[key] is not None]
    return float(np.mean(v)) if v else None


def aggregate(rows):
    return {
        "states": len(rows),
        **{
            key: mean_present(rows, key)
            for key in (
                "top1_accuracy",
                "best_action_hit_rate",
                "pairwise_accuracy",
                "spearman",
                "kendall_tau_b",
                "regret_yuan",
                "chosen_EV_yuan",
                "resolved_pairwise_accuracy",
            )
        },
        "non_tied_pairs": sum(r["non_tied_pairs"] for r in rows),
        "pooled_pairwise_accuracy": sum(r["pairwise_correct_sum"] for r in rows)
        / sum(r["non_tied_pairs"] for r in rows)
        if sum(r["non_tied_pairs"] for r in rows)
        else None,
        "regret_distribution_yuan": summary([r["regret_yuan"] for r in rows]),
        "margin_pearson": correlation(
            [r["model_margin_native"] for r in rows], [r["rollout_margin_yuan"] for r in rows]
        ),
        "margin_spearman": correlation(
            ranks([r["model_margin_native"] for r in rows]),
            ranks([r["rollout_margin_yuan"] for r in rows]),
        )
        if rows
        else None,
    }


def summarize_results(results):
    strategies = [
        "baseline",
        *[
            f"{cp}/{mode}"
            for cp in ("initial", "long10", "A10", "B10")
            for mode in ("raw", "anchor")
        ],
    ]
    output = {}
    for strategy in strategies:
        output[strategy] = {}
        for family in ("all", *QUOTAS):
            for subset in ("all", "clear", "ambiguous"):
                roots = [
                    r
                    for r in results
                    if (family == "all" or r["family"] == family)
                    and (subset == "all" or r["ev"]["clear"] == (subset == "clear"))
                ]
                output[strategy][f"{family}:{subset}"] = aggregate(
                    [r["ranking"][strategy] for r in roots]
                )
        output[strategy]["calibration"] = {}
        bounds = (
            [0, 0.5, 1, 2, 5, 10, 25, math.inf]
            if strategy.endswith("/raw")
            else [0, 5, 20, 50, 100, 200, math.inf]
        )
        for lo, hi in pairwise(bounds):
            # Only raw Q is in cash units. Anchor score includes a log-prior.
            scale = 1 if strategy == "baseline" else 100
            selected = [
                r["ranking"][strategy]
                for r in results
                if lo <= r["ranking"][strategy]["model_margin_native"] * scale < hi
            ]
            output[strategy]["calibration"][f"{lo}:{hi}"] = aggregate(selected)
    return output


def policy_bundle(models, baseline, behavior):
    for model in [*models.values(), baseline]:
        model.eval().requires_grad_(False)
    policy = ExtractionPolicy(
        models["initial"], baseline, behavior, {"mode": "anchor", "lambda": 0.3}
    )
    policy.eval().requires_grad_(False)
    return policy


def assert_frozen(models, before):
    assert {name: parameter_hash(m) for name, m in models.items()} == before
    assert all(
        not m.training and all(p.grad is None for p in m.parameters()) for m in models.values()
    )
