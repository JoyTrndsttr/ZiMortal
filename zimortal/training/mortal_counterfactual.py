"""Finite, diagnostic-only high-support paired frozen-baseline rollouts."""

import argparse
import gzip
import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from zimortal.engine import GameState, RuleClarificationRequired, RuleEngine
from zimortal.model.encoding import encode_action, encode_observation

from .active import atomic_json
from .mortal_data import load_rows, write_bytes
from .mortal_diagnostics import distribution
from .mortal_offline import BASELINE
from .mortal_v2 import game_split
from .online_rl import digest, infer, source_hashes
from .rollout_engine import RolloutEngine
from .runtime import load_model

FAMILIES = ("discard", "chi/pass", "peng/pass", "hu/pass")


def state_key(features, actions):
    return hashlib.sha256(features.tobytes() + actions.tobytes()).hexdigest()


class Frequencies:
    """Empirical TRAIN-only counts of network inputs, not behavior probabilities."""

    def __init__(self, train):
        self.rows = len(train)
        self.eligible, self.chosen = Counter(), Counter()
        self.family_eligible, self.family_chosen = Counter(), Counter()
        self.states, self.state_actions = Counter(), Counter()
        for row in train:
            keys = [a.tobytes() for a in row["actions"]]
            key = state_key(row["features"], row["actions"])
            selected = keys[int(row["chosen"])]
            self.eligible.update(set(keys))
            self.chosen[selected] += 1
            family = str(row.get("family", "unknown"))
            self.family_eligible.update((family, k) for k in set(keys))
            self.family_chosen[family, selected] += 1
            self.states[key] += 1
            self.state_actions[key, selected] += 1

    def lookup(self, features, actions, index, family=None):
        key = actions[index].tobytes()
        state = state_key(features, actions)
        result = {
            "chosen_training_rows": self.chosen[key],
            "eligible_training_rows": self.eligible[key],
            "chosen_given_eligible": self.chosen[key] / self.eligible[key]
            if self.eligible[key]
            else None,
            "chosen_fraction_all_train": self.chosen[key] / self.rows,
            "exact_input_training_rows": self.states[state],
            "exact_input_chosen_rows": self.state_actions[state, key],
        }
        if family is not None:
            result.update(
                family_chosen_training_rows=self.family_chosen[str(family), key],
                family_eligible_training_rows=self.family_eligible[str(family), key],
            )
        return result


def sample_roots(population, per_family=100, seed=1416):
    rng = np.random.default_rng(seed)
    selected = []
    for family in FAMILIES:
        rows = sorted(
            [r for r in population if r["family"] == family],
            key=lambda r: (r["pair_q_gap"], r["seed"], r["seat"], r["step"]),
        )
        if not rows:
            continue
        count = min(per_family, len(rows))
        strata = np.array_split(np.arange(len(rows)), 4)
        allocation = [count // 4 + (i < count % 4) for i in range(4)]
        for i, (indices, take) in enumerate(zip(strata, allocation, strict=True)):
            for index in sorted(rng.choice(indices, take, replace=False).tolist()):
                selected.append(
                    {
                        **rows[index],
                        "sampling_stratum": i,
                        "inclusion_probability": take / len(indices),
                        "sampling_weight": len(indices) / take,
                    }
                )
    return selected


def particle(root, seed):
    out = root.clone()
    np.random.default_rng(seed).shuffle(out.deck)
    out.validate()
    return out


def paired_rollouts(
    root, parent_action, selected_action, model, *, count=32, seed=1416, device="cpu"
):
    """Same fixed hidden hands, paired uniform remaining-deck permutation.

    Only Observation enters the frozen greedy continuation network. Original
    hidden hands are diagnostic conditioning, never search-policy information.
    """
    engine = RolloutEngine()
    if parent_action == selected_action or count < 2:
        raise ValueError("distinct actions and at least two paired particles required")
    original = root.serialize()
    roots, states, seeds = [], [], []
    for i in range(count):
        common_seed = int(np.random.SeedSequence([seed, i]).generate_state(1, dtype=np.uint64)[0])
        world = particle(root, common_seed)
        roots.append(world)
        seeds.append(common_seed)
        for action in (parent_action, selected_action):
            state = engine.step(world, action)
            state.validate()
            states.append(state)
    for _ in range(1000):
        waiting, observations = [], []
        for index, state in enumerate(states):
            if state.terminal:
                continue
            actions = engine.legal_actions(state)
            if len(actions) == 1:
                states[index] = engine.step(state, actions[0])
                states[index].validate()
            else:
                waiting.append(index)
                observations.append(engine.observation(state, actions[0].player))
        if all(s.terminal for s in states):
            break
        for start in range(0, len(waiting), 32):
            obs = observations[start : start + 32]
            if not obs:
                continue
            choices = infer(model, obs, device)[0]
            for index, visible, choice in zip(
                waiting[start : start + 32], obs, choices, strict=True
            ):
                states[index] = engine.step(states[index], visible.legal_actions[choice])
                states[index].validate()
    else:
        raise RuntimeError("counterfactual continuation exceeded 1000 iterations")
    evidence, outcomes = [], []
    ordinary = RuleEngine()
    for i, (world, common_seed) in enumerate(zip(roots, seeds, strict=True)):
        terminals = states[2 * i : 2 * i + 2]
        for end in terminals:
            end.validate()
            rebuilt = ordinary.replay(world, end.history[len(world.history) :])
            if rebuilt.serialize() != end.serialize():
                raise RuntimeError("paired suffix replay mismatch")
        cash = [
            end.settlement.payments[parent_action.player] if end.settlement else 0
            for end in terminals
        ]
        outcomes.append(cash)
        evidence.append(
            {
                "common_seed": common_seed,
                "deck": world.deck,
                "parent_terminal": terminals[0].serialize(),
                "selected_terminal": terminals[1].serialize(),
            }
        )
    if root.serialize() != original:
        raise RuntimeError("counterfactual mutated original root")
    return np.asarray(outcomes, dtype=float), evidence


def paired_statistics(outcomes, seed=1416):
    values = np.asarray(outcomes, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("finite complete paired outcomes required")
    delta = values[:, 1] - values[:, 0]
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(len(delta), size=(4096, len(delta)))].mean(-1)
    ci = np.quantile(means, [0.025, 0.975]).tolist()
    return {
        "rollouts_per_action": len(values),
        "parent_mean_cash": float(values[:, 0].mean()),
        "selected_mean_cash": float(values[:, 1].mean()),
        "paired_delta": float(delta.mean()),
        "paired_standard_error": float(delta.std(ddof=1) / np.sqrt(len(delta))),
        "paired_bootstrap_95": ci,
        "sign": "positive"
        if delta.mean() > 0
        else "negative"
        if delta.mean() < 0
        else "sample_zero",
        "resolved_sign": "positive" if ci[0] > 0 else "negative" if ci[1] < 0 else "unresolved",
        "halves_delta": [float(a.mean()) for a in np.array_split(delta, 2)],
    }


def ranks(values):
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    del unique
    middle = np.cumsum(counts) - (counts - 1) / 2
    return middle[inverse]


def correlation(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return (
        float(np.corrcoef(a, b)[0, 1])
        if len(a) > 1 and a.std() > 1e-10 and b.std() > 1e-10
        else None
    )


def ranking_summary(rows, seed=1416):
    if not rows:
        return {"roots": 0, "reason": "no eligible changed states; no fabricated accuracy"}
    weights = np.array([r["sampling_weight"] for r in rows])
    w = weights / weights.sum()
    delta = np.array([r["stats"]["paired_delta"] for r in rows])
    gap = np.array([r["pair_q_gap"] for r in rows])
    bootstrap = np.random.default_rng(seed)
    groups = sorted({r["seed"] for r in rows})
    nums = np.array(
        [sum(weights[i] * delta[i] for i, r in enumerate(rows) if r["seed"] == g) for g in groups]
    )
    dens = np.array([sum(weights[i] for i, r in enumerate(rows) if r["seed"] == g) for g in groups])
    indices = bootstrap.integers(len(groups), size=(4096, len(groups)))
    means = nums[indices].sum(-1) / dens[indices].sum(-1)
    resolved = [r for r in rows if r["stats"]["resolved_sign"] != "unresolved"]
    nonzero = w[delta != 0].sum()
    return {
        "roots": len(rows),
        "game_seeds": len(groups),
        "weighted_mean_parent_cash": float(
            sum(a * r["stats"]["parent_mean_cash"] for a, r in zip(w, rows, strict=True))
        ),
        "weighted_mean_selected_cash": float(
            sum(a * r["stats"]["selected_mean_cash"] for a, r in zip(w, rows, strict=True))
        ),
        "weighted_mean_delta": float(w @ delta),
        "seed_cluster_bootstrap_95": np.quantile(means, [0.025, 0.975]).tolist(),
        "positive_fraction": float(w[delta > 0].sum()),
        "negative_fraction": float(w[delta < 0].sum()),
        "sample_zero_fraction": float(w[delta == 0].sum()),
        "nonzero_ranking_accuracy": float(w[delta > 0].sum() / nonzero) if nonzero else None,
        "resolved_positive_roots": sum(r["stats"]["resolved_sign"] == "positive" for r in resolved),
        "resolved_negative_roots": sum(r["stats"]["resolved_sign"] == "negative" for r in resolved),
        "unresolved_roots": len(rows) - len(resolved),
        "pair_gap_pearson": correlation(gap, delta),
        "pair_gap_spearman": correlation(ranks(gap), ranks(delta)),
        "retained_margin_pearson": correlation([r["margin"] * 100 for r in rows], delta),
        "retained_margin_spearman": correlation(
            ranks([r["margin"] * 100 for r in rows]), ranks(delta)
        ),
        "paired_se": distribution([r["stats"]["paired_standard_error"] for r in rows]),
        "half_sign_agreement": float(
            np.mean(
                [
                    np.sign(r["stats"]["halves_delta"][0]) == np.sign(r["stats"]["halves_delta"][1])
                    for r in rows
                ]
            )
        ),
    }


def prepare(root, source, frequencies):
    indexed = defaultdict(dict)
    with gzip.open(source / "screening/decisions-cql-support.jsonl.gz", "rt") as file:
        for line in file:
            record = json.loads(line)
            if (
                record["changed"]
                and min(record["behavior_probability"], record["parent_probability"]) >= 0.1
            ):
                indexed[record["seed"], record["seat"]][record["step"]] = record
    population, states = [], {}
    engine = RuleEngine()
    with gzip.open(source / "screening/screen-cql-support-replays.jsonl.gz", "rt") as file:
        for line in file:
            evidence = json.loads(line)
            wanted = indexed.pop((evidence["seed"], evidence["seat"]), {})
            if not wanted:
                continue
            end = GameState.deserialize(evidence["state"])
            state = engine.new_game(evidence["seed"], evidence["dealer"])
            for step, action in enumerate(end.history, 1):
                if step in wanted:
                    r = wanted.pop(step)
                    obs = engine.observation(state, evidence["seat"])
                    legal = obs.legal_actions
                    if (
                        json.loads(json.dumps([asdict(a) for a in legal])) != r["actions"]
                        or legal[r["chosen"]] != action
                    ):
                        raise RuntimeError("diagnostic root legal actions/history mismatch")
                    x = encode_observation(obs, "huxi")
                    a = np.stack([encode_action(v, obs.player) for v in legal])
                    r = {
                        **r,
                        "id": f"{r['seed']}-{r['seat']}-{step}",
                        "q_parent": r["q"][r["parent"]] * 100,
                        "q_selected": r["q"][r["chosen"]] * 100,
                        "pair_q_gap": (r["q"][r["chosen"]] - r["q"][r["parent"]]) * 100,
                        "high_support_candidates": sum(p >= 0.1 for p in r["probabilities"]),
                        "direction": r["actions"][r["parent"]]["kind"]
                        + "->"
                        + r["actions"][r["chosen"]]["kind"],
                        "parent_frequency": frequencies.lookup(x, a, r["parent"], r["family"]),
                        "selected_frequency": frequencies.lookup(x, a, r["chosen"], r["family"]),
                    }
                    population.append(r)
                    states[r["id"]] = state.serialize()
                state = engine.step(state, action)
            if wanted or state.serialize() != end.serialize():
                raise RuntimeError("incomplete root reconstruction")
    if indexed:
        raise RuntimeError("unmatched diagnostic roots")
    atomic_json(root / "population.json", population)
    return population, states


def population_summary(rows):
    return {
        family: {
            "states": len(group),
            "directions": dict(Counter(r["direction"] for r in group)),
            "parent_probability": distribution([r["parent_probability"] for r in group]),
            "selected_probability": distribution([r["behavior_probability"] for r in group]),
            "q_parent": distribution([r["q_parent"] for r in group]),
            "q_selected": distribution([r["q_selected"] for r in group]),
            "pair_q_gap": distribution([r["pair_q_gap"] for r in group]),
            "retained_margin_cash": distribution([r["margin"] * 100 for r in group]),
            **{
                f"{who}_{key}": distribution(
                    [r[f"{who}_frequency"][key] for r in group if key in r[f"{who}_frequency"]]
                )
                for who in ("parent", "selected")
                for key in (
                    "chosen_training_rows",
                    "eligible_training_rows",
                    "exact_input_training_rows",
                    "exact_input_chosen_rows",
                    "family_chosen_training_rows",
                    "family_eligible_training_rows",
                )
            },
        }
        for family in FAMILIES
        for group in [[r for r in rows if r["family"] == family]]
    }


def run(args):
    torch.set_num_threads(2)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no silent fallback")
    root, source = Path(args.directory), Path(args.source)
    root.mkdir(parents=True, exist_ok=False)
    lock = json.loads((source / "selection-lock.json").read_text())
    if lock["alpha"] != 0.1 or lock["support_threshold"] != 0.1:
        raise RuntimeError("fixed CQL/support settings required")
    q_path = source / lock["cql_variant"] / "best-dev.pt"
    expected_q_sha = next(
        sha
        for path, sha in lock["checkpoints"].items()
        if Path(path).parent.name == lock["cql_variant"]
    )
    if digest(q_path) != expected_q_sha:
        raise RuntimeError("frozen Q checkpoint changed")
    data = Path("data/generated/mortal-v2/data")
    manifest = json.loads((data / "manifest.json").read_text())
    original = json.loads((source / "config.json").read_text())
    if (
        digest(BASELINE) != original["baseline_sha256"]
        or digest(data / "manifest.json") != original["manifest_sha256"]
    ):
        raise RuntimeError("baseline/data changed")
    inputs = {
        str(source / "screening" / name): digest(source / "screening" / name)
        for name in ("decisions-cql-support.jsonl.gz", "screen-cql-support-replays.jsonl.gz")
    }
    config = {
        **vars(args),
        "source_hashes": source_hashes(),
        "input_sha256": inputs,
        "checkpoint_sha256": {BASELINE: digest(BASELINE), str(q_path): digest(q_path)},
        "definition": "fixed true root hands/history; uniform remaining-deck permutations; paired first actions; all subsequent actions frozen baseline greedy",
        "not_for_training": True,
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
    }
    atomic_json(root / "config.json", config)
    train, _, _ = game_split(load_rows(data, manifest))
    frequencies = Frequencies(train)
    population, states = prepare(root, source, frequencies)
    samples = sample_roots(population, args.per_family, args.seed)
    atomic_json(root / "sampling.json", samples)
    atomic_json(root / "population-diagnostics.json", population_summary(population))
    baseline = load_model(BASELINE, args.device).requires_grad_(False)
    successes, failures = [], []
    started = time.perf_counter()
    for index, row in enumerate(samples):
        state = GameState.deserialize(states[row["id"]])
        legal = RuleEngine().legal_actions(state)
        rng_seed = int.from_bytes(
            hashlib.sha256(f"{args.seed}:{row['id']}".encode()).digest()[:4], "little"
        )
        try:
            outcomes, evidence = paired_rollouts(
                state,
                legal[row["parent"]],
                legal[row["chosen"]],
                baseline,
                count=args.rollouts,
                seed=rng_seed,
                device=args.device,
            )
            result = {
                **row,
                "stats": paired_statistics(outcomes, rng_seed),
                "distinct_decks": len({tuple(p["deck"]) for p in evidence}),
                "outcomes": outcomes.tolist(),
            }
            path = root / f"root-{index:04d}.json.gz"
            sha = write_bytes(
                path,
                gzip.compress(
                    json.dumps(
                        {"root": state.serialize(), "result": result, "pairs": evidence}
                    ).encode(),
                    mtime=0,
                ),
            )
            successes.append({**result, "evidence": path.name, "sha256": sha})
        except RuleClarificationRequired as exc:
            failure = {
                "root": row,
                "state": state.serialize(),
                "error": str(exc),
                "whole_root_excluded": True,
                "common_seed_base": rng_seed,
                "rollouts": args.rollouts,
            }
            atomic_json(root / f"failure-{index:04d}.json", failure)
            failures.append(failure)
        atomic_json(
            root / "progress.json",
            {
                "status": "diagnostic_rollouts",
                "roots_done": index + 1,
                "roots_total": len(samples),
                "valid": len(successes),
                "failed": len(failures),
                "seconds": time.perf_counter() - started,
            },
        )
    atomic_json(root / "roots.json", successes)
    summary = {
        "status": "complete_counterfactual_diagnostic_stop",
        "train_rows": len(train),
        "population": {f: sum(r["family"] == f for r in population) for f in FAMILIES},
        "sampled": len(samples),
        "valid": len(successes),
        "failures": failures,
        "seconds": time.perf_counter() - started,
        "population_diagnostics": population_summary(population),
        "families": {
            f: ranking_summary([r for r in successes if r["family"] == f]) for f in FAMILIES
        },
        "overall": ranking_summary(successes),
        "not_for_training": True,
    }
    summary["bins"] = {
        name: {
            f"{lo}:{hi}": ranking_summary([r for r in successes if lo <= r[field] < hi])
            for lo, hi in bins
        }
        for name, field, bins in (
            ("pair_q_gap", "pair_q_gap", ((0, 2), (2, 5), (5, 10), (10, 1000))),
            ("retained_margin", "margin", ((0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 1000))),
            (
                "selected_probability",
                "behavior_probability",
                ((0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 1.01)),
            ),
        )
    }
    if digest(data / "manifest.json") != original["manifest_sha256"]:
        raise RuntimeError("data manifest changed")
    for shard in manifest["shards"]:
        if digest(data / shard["data"]) != shard["sha256"]:
            raise RuntimeError("training shard changed")
    truth = {r["id"]: r["stats"] for r in successes}
    atomic_json(
        root / "annotated-population.json",
        [
            {**r, "rollout_diagnostic": truth.get(r["id"], {"sign": "not_measured"})}
            for r in population
        ],
    )
    if source_hashes() != config["source_hashes"] or any(
        digest(p) != sha for p, sha in (config["checkpoint_sha256"] | inputs).items()
    ):
        raise RuntimeError("frozen producer/models/evidence changed")
    atomic_json(root / "result.json", summary)
    atomic_json(root / "progress.json", {"status": summary["status"]})
    print(summary["status"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="data/generated/mortal-counterfactual-v1")
    parser.add_argument("--source", default="data/generated/mortal-support-v1")
    parser.add_argument("--per-family", type=int, default=100)
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1416)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if Path(args.directory).exists() or args.per_family < 4 or args.rollouts < 2:
        parser.error("new directory, >=4 roots/family and >=2 paired rollouts required")
    try:
        run(args)
    except Exception as exc:
        root = Path(args.directory)
        if root.exists():
            atomic_json(root / "progress.json", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
