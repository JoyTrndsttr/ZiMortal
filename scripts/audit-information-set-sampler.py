"""Independent prefix replay, truth-support (audit only), and sampler statistics.

No game is continued beyond its locked root. No models are instantiated.
"""

import argparse
import gzip
import importlib.util
import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from math import factorial
from pathlib import Path

from zimortal.belief.conditional import HistorySampler, satisfies_discard_passes
from zimortal.belief.sampling import observed_draws, reverse_hands
from zimortal.engine import GameState, PlayerState, RuleEngine
from zimortal.engine.scoring import meld_huxi
from zimortal.training.information_set_sampler import (
    HiddenWorld,
    InformationSetSampler,
    PublicInformationState,
    SlotPrior,
    enumerate_physical_worlds,
    observation_bytes,
    physical_pool,
    public_from_json,
)


@dataclass(frozen=True)
class AuditOnlyTruth:
    """Never accepted by InformationSetSampler, nor passed to generation workers."""

    full_state: GameState


def runner():
    spec = importlib.util.spec_from_file_location(
        "information_set_runner", "scripts/sample-information-set-worlds.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summary(values):
    values = sorted(values)
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": values[0],
        "max": values[-1],
        **{f"p{p}": values[round((len(values) - 1) * p / 100)] for p in (5, 25, 50, 75, 95)},
    }


def diversity(counts):
    n = sum(counts.values())
    return {
        "unique": len(counts),
        "empirical_entropy_bits": -sum(c / n * math.log2(c / n) for c in counts.values()),
        "effective_unique": n * n / sum(c * c for c in counts.values()),
        "largest_empirical_mass": max(counts.values()) / n,
    }


def marginal(hist, n):
    return [
        {
            "tile": tile,
            "count_probability": [hist[tile][i] / n for i in range(5)],
            "present_probability": 1 - hist[tile][0] / n,
            "mean_count": sum(i * hist[tile][i] for i in range(5)) / n,
        }
        for tile in range(20)
    ]


def independent_replay(public, world):
    """Use the older reverse implementation, not the generator's reconstruction."""
    obs = public.observation
    draws = observed_draws(obs)
    original = reverse_hands(obs, [list(h) for h in world.hands], draws)
    assert all(len(h) == (21 if p == public.dealer else 20) for p, h in enumerate(original))
    initial = GameState(
        [
            PlayerState(sorted(h), kans={t for t, n in Counter(h).items() if n == 3})
            for h in original
        ],
        list(world.wall) + list(reversed(draws)),
        dealer=public.dealer,
        turn=public.dealer,
        phase="opening",
    )
    initial.validate()
    engine = RuleEngine()
    state = engine.replay(initial, obs.history)
    state.validate()
    assert observation_bytes(engine.observation(state, obs.player)) == observation_bytes(obs)
    assert HiddenWorld.from_state(state) == world
    return state


def toy_audit():
    results = []
    for number in (82, 155):
        path = Path(f"tests/fixtures/information-set/small-{number}.json")
        public = public_from_json(json.loads(path.read_text()))
        sampler = InformationSetSampler(public)
        legal, illegal = {}, 0
        for world in enumerate_physical_worlds(public):
            try:
                independent_replay(public, world)
            except ValueError:
                illegal += 1
                continue
            assert sampler.contains(world)
            # Ordered-wall mass is proportional to 1 / prod(a! b!).
            denominator = math.prod(
                factorial(n) for p in sampler.opponents for n in Counter(world.hands[p]).values()
            )
            legal[world.key] = Fraction(1, denominator)
        total = sum(legal.values())
        expected = {key: float(mass / total) for key, mass in legal.items()}
        counts = Counter(sampler.sample(seed)[0].key for seed in range(512))
        assert set(counts) == set(legal)
        tv = 0.5 * sum(abs(counts[key] / 512 - p) for key, p in expected.items())
        assert tv < 0.15  # Fixed loose finite-N sanity bound, not a universal proof.
        results.append(
            {
                "fixture": str(path),
                "physical_assignments": len(legal) + illegal,
                "legal_support": len(legal),
                "illegal_support": illegal,
                "sampler_covered": len(counts),
                "sample_count": 512,
                "empirical_tv": tv,
                "probabilities": expected,
            }
        )
    return results


def id_invariance(public):
    pool, _, sizes = physical_pool(public)
    physical = [(tile, copy) for tile, n in enumerate(pool) for copy in range(n)]
    renamed = [(tile, 700 + 3 - copy) for tile, copy in reversed(physical)]
    streams = []
    for labeled in (physical, renamed):
        counts = Counter(tile for tile, _ in labeled)
        collapsed = tuple(counts[t] for t in range(20))
        assert collapsed == pool
        prior = SlotPrior(collapsed, sizes)
        streams.append([prior.draw(random.Random(seed)) for seed in range(16)])
    assert streams[0] == streams[1]
    # Physical IDs are absent from the engine and Observation; only type counts exist.
    return True


def local_allocation_count(prior):
    from functools import lru_cache

    @lru_cache(None)
    def count(tile, a_left, b_left):
        if min(a_left, b_left) < 0:
            return 0
        if tile == len(prior.pool):
            return int(a_left == b_left == 0)
        return sum(count(tile + 1, a_left - a, b_left - b) for a, b, _, _ in prior.options[tile])

    return count(0, *prior.sizes)


def audit_root(directory, record, config):
    helpers = runner()
    root_id = record["id"]
    path = directory / "public" / f"{root_id}.json"
    assert helpers.digest(path) == record["public_sha256"]
    public = public_from_json(json.loads(path.read_text()))
    assert public.sha256 == record["observation_sha256"]
    sampler = InformationSetSampler(public)
    old = HistorySampler(public.observation)
    assert helpers.digest(record["audit_source"]) == record["audit_source_sha256"]
    with gzip.open(record["audit_source"], "rt") as file:
        raw_truth = json.load(file)
    truth = AuditOnlyTruth(GameState.deserialize(raw_truth["root"]))
    # No future suffix/initial hidden deal is read or used in generation.
    assert observation_bytes(
        RuleEngine().observation(truth.full_state, public.observation.player)
    ) == observation_bytes(public.observation)
    truth_world = HiddenWorld.from_state(truth.full_state)
    assert sampler.contains(truth_world), f"true world missing: {root_id}"
    old_true_pass = satisfies_discard_passes(truth_world.hands, old.pass_checks)
    hand_counts = [Counter(), Counter()]
    joint_counts, world_counts, wall_counts, wall_multisets = (
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    )
    held = [[Counter() for _ in range(20)] for _ in range(2)]
    wall = [Counter() for _ in range(20)]
    next_draw = Counter()
    times, attempts, rejections = [], [], Counter()
    witnesses = []
    old_filter_rejections = 0
    world_file = directory / "worlds" / f"{root_id}.jsonl.gz"
    sampling = json.loads((directory / f"{root_id}-sampling.json").read_text())
    assert helpers.digest(world_file) == sampling["sha256"]
    with gzip.open(world_file, "rt") as file:
        for index, line in enumerate(file):
            row = json.loads(line)
            assert row["index"] == index and row["seed"] == helpers.seeded(root_id, index)
            world = HiddenWorld(
                tuple(tuple(h) for h in row["world"]["hands"]), tuple(row["world"]["wall"])
            )
            state = independent_replay(public, world)
            obs = public.observation
            # Observation byte equality already checks public sizes/actions/melds.
            own = state.players[obs.player]
            true_own = truth.full_state.players[obs.player]
            assert own.kans == true_own.kans
            assert sum(meld_huxi(m) for m in own.melds) + sum(
                meld_huxi_kind_kan(t) for t in own.kans
            ) == sum(meld_huxi(m) for m in true_own.melds) + sum(
                meld_huxi_kind_kan(t) for t in true_own.kans
            )
            if len(witnesses) < 2 and (not witnesses or world.hands != witnesses[0].hands):
                witnesses.append(world)
            old_filter_rejections += not satisfies_discard_passes(world.hands, old.pass_checks)
            for j, p in enumerate(sampler.opponents):
                hand_counts[j][world.hands[p]] += 1
                counts = Counter(world.hands[p])
                for tile in range(20):
                    held[j][tile][counts[tile]] += 1
            joint_counts[tuple(world.hands[p] for p in sampler.opponents)] += 1
            world_counts[world.key] += 1
            wall_counts[world.wall] += 1
            wall_multisets[tuple(sorted(world.wall))] += 1
            counts = Counter(world.wall)
            for tile in range(20):
                wall[tile][counts[tile]] += 1
            if world.wall:
                next_draw[world.wall[-1]] += 1
            assert row["attempts"] == 1 + sum(row["rejections"].values())
            attempts.append(row["attempts"])
            rejections.update(row["rejections"])
            times.append(row["seconds"])
    assert len(times) == config["worlds_per_root"]
    assert sum(attempts) == sampling["proposals"] and dict(rejections) == sampling["rejections"]
    # Reproduce actual sampler output, not just a distributional comparison.
    for index in (0, 999):
        actual, a, r = sampler.sample(
            helpers.seeded(root_id, index), config["max_proposals_per_world"]
        )
        assert actual.key in world_counts
        if index == 0:
            assert actual == witnesses[0]
        assert a >= 1 and a == sum(r.values()) + 1
    leakage = "no distinct-hand witness in fixed sample"
    if len(witnesses) == 2:
        projected = [
            PublicInformationState(
                RuleEngine().observation(independent_replay(public, w), obs.player)
            )
            for w in witnesses
        ]
        assert projected[0] == projected[1] == public
        paired = [InformationSetSampler(p) for p in projected]
        assert paired[0].sample(helpers.seeded(root_id, 1001)) == paired[1].sample(
            helpers.seeded(root_id, 1001)
        )
        leakage = "identical same-seed output from distinct true hands"
    assert id_invariance(public)
    result = {
        "id": root_id,
        "family": record["family"],
        "accepted": len(times),
        "certified_replays": len(times),
        "truth_support": True,
        "old_true_pass_prefilter_support": old_true_pass,
        "old_prefilter_rejects_new_legal_samples": old_filter_rejections,
        "proposals": sum(attempts),
        "proposal_acceptance": len(times) / sum(attempts),
        "rejections": dict(rejections),
        "attempts": summary(attempts),
        "sample_seconds": summary(times),
        "hand_diversity": [diversity(c) for c in hand_counts],
        "joint_hand_diversity": diversity(joint_counts),
        "world_diversity": diversity(world_counts),
        "ordered_wall_diversity": diversity(wall_counts),
        "wall_multiset_diversity": diversity(wall_multisets),
        "opponent_marginals": [marginal(h, len(times)) for h in held],
        "wall_marginals": marginal(wall, len(times)),
        "next_draw_probability": {str(t): n / len(times) for t, n in next_draw.items()},
        "local_count_allocation_upper_bound": local_allocation_count(sampler.prior),
        "leakage": leakage,
        "id_permutation_invariance": True,
    }
    helpers.save(directory / f"{root_id}-audit.json", result)
    print(
        "AUDITED",
        root_id,
        round(result["proposal_acceptance"], 4),
        result["joint_hand_diversity"]["unique"],
        flush=True,
    )
    return result


def meld_huxi_kind_kan(tile):
    return 3 if tile < 10 else 6


def run(directory):
    helpers = runner()
    config = json.loads((directory / "config.json").read_text())
    lock = json.loads((directory / "lock.json").read_text())
    assert helpers.digest(directory / "config.json") == lock["config_sha256"]
    assert helpers.digest(directory / "locked-root-manifest.json") == lock["manifest_sha256"]
    assert helpers.frozen_sources() == config["sources"]
    assert all(helpers.digest(p) == h for p, h in config["checkpoints"].items())
    manifest = json.loads((directory / "locked-root-manifest.json").read_text())
    assert Counter(r["family"] for r in manifest) == Counter(config["quotas"])
    started = time.perf_counter()
    toys = toy_audit()
    results = [audit_root(directory, r, config) for r in manifest]
    assert helpers.frozen_sources() == config["sources"]
    assert all(helpers.digest(p) == h for p, h in config["checkpoints"].items())
    flags = [
        r["id"]
        for r in results
        if r["joint_hand_diversity"]["effective_unique"] < 10
        and r["local_count_allocation_upper_bound"] > 1000
    ]
    report = {
        "roots": results,
        "toy_exhaustive": toys,
        "collapse_flags": flags,
        "total_samples": sum(r["accepted"] for r in results),
        "total_certified_replays": sum(r["certified_replays"] for r in results),
        "total_proposals": sum(r["proposals"] for r in results),
        "truth_in_support_roots": sum(r["truth_support"] for r in results),
        "distinct_hand_leakage_roots": sum(r["leakage"].startswith("identical") for r in results),
        "old_prefilter_rejections": sum(
            r["old_prefilter_rejects_new_legal_samples"] for r in results
        ),
        "sampling_seconds": summary([r["sample_seconds"]["mean"] for r in results]),
        "joint_hand_effective": summary(
            [r["joint_hand_diversity"]["effective_unique"] for r in results]
        ),
        "joint_hand_entropy": summary(
            [r["joint_hand_diversity"]["empirical_entropy_bits"] for r in results]
        ),
        "joint_hand_unique": summary([r["joint_hand_diversity"]["unique"] for r in results]),
        "world_effective": summary([r["world_diversity"]["effective_unique"] for r in results]),
        "audit_seconds": time.perf_counter() - started,
        "training_updates": 0,
        "future_rollouts": 0,
        "certification": "PASS within declared Observation contract and tested support; NOT exact Bayesian posterior"
        if not flags
        else "REVIEW collapse flags",
        "config_sha256": helpers.digest(directory / "config.json"),
        "root_manifest_sha256": helpers.digest(directory / "locked-root-manifest.json"),
    }
    helpers.save(directory / "audit-result.json", report)
    outputs = {str(p): helpers.digest(p) for p in sorted(directory.glob("worlds/*.gz"))}
    outputs[str(directory / "audit-result.json")] = helpers.digest(directory / "audit-result.json")
    helpers.save(directory / "output-sha256.json", outputs)
    print(report["certification"], report["total_samples"], report["total_proposals"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, default=Path("data/generated/information-set-sampler-v1")
    )
    run(parser.parse_args().directory)
