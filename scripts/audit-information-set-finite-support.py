"""Supplemental exact support audit of ALL publicly small finite locked roots.

Post-sampling sanity only, never change root selection, sampler, or weights.
No new sampling seeds, action rollouts, policies, training, or hidden truth inputs.
"""

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from fractions import Fraction
from pathlib import Path

from zimortal.training.information_set_sampler import (
    InformationSetSampler,
    canonical,
    certify_world,
    enumerate_physical_worlds,
    physical_pool,
    public_from_json,
)


def run(directory):
    report = []
    inputs = {}
    rng_checks = 0
    for record in json.loads((directory / "locked-root-manifest.json").read_text()):
        path = directory / "public" / f"{record['id']}.json"
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == record["public_sha256"]
        public = public_from_json(json.loads(raw))
        worlds_path = directory / "worlds" / f"{record['id']}.jsonl.gz"
        inputs[str(path)] = record["public_sha256"]
        inputs[str(worlds_path)] = hashlib.sha256(worlds_path.read_bytes()).hexdigest()
        sampler = InformationSetSampler(public)
        with gzip.open(worlds_path, "rt") as file:
            for index, line in enumerate(file):
                if index not in (0, 999):
                    continue
                row = json.loads(line)
                world, attempts, rejects = sampler.sample(row["seed"])
                assert world.key == canonical(row["world"])
                assert attempts == row["attempts"] and rejects == row["rejections"]
                rng_checks += 1
        pool, opponents, sizes = physical_pool(public)
        n, a, b = sum(pool), *sizes
        d = public.observation.remaining_tiles
        upper = math.factorial(n) // (math.factorial(a) * math.factorial(b) * math.factorial(d))
        # Fixed computational eligibility, based solely on public sizes.
        if d > 1 or upper > 10000:
            continue
        legal, physical = {}, 0
        for world in enumerate_physical_worlds(public, 10000):
            physical += 1
            try:
                certify_world(public, world)
            except ValueError:
                continue
            den = math.prod(
                math.factorial(c) for p in opponents for c in Counter(world.hands[p]).values()
            )
            legal[world.key] = Fraction(1, den)
        observed = Counter()
        worlds_path = directory / "worlds" / f"{record['id']}.jsonl.gz"
        inputs[str(path)] = record["public_sha256"]
        inputs[str(worlds_path)] = hashlib.sha256(worlds_path.read_bytes()).hexdigest()
        with gzip.open(worlds_path, "rt") as file:
            for line in file:
                observed[canonical(json.loads(line)["world"])] += 1
        assert set(observed) == set(legal)
        z, count = sum(legal.values()), sum(observed.values())
        expected = {key: float(mass / z) for key, mass in legal.items()}
        tv = 0.5 * sum(abs(observed[key] / count - prob) for key, prob in expected.items())
        row = {
            "id": record["id"],
            "physical_worlds": physical,
            "legal_support": len(legal),
            "sampled_support": len(observed),
            "expected_probability": expected,
            "observed_frequency": dict(observed),
            "empirical_tv": tv,
        }
        report.append(row)
        print(record["id"], physical, len(legal), "TV", tv, flush=True)
    result = {
        "selection": "SUPPLEMENTAL after sampling: ALL locked roots with wall<=1 and labeled physical slot split bound<=10000; no cash/Q-based selection, new sampling seeds or future steps; existing seeds regenerated only for RNG audit",
        "roots": report,
        "exact_rng_reproductions": rng_checks,
        "rng_indices_per_root": [0, 999],
        "inputs": inputs,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (directory / "finite-support-check.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, default=Path("data/generated/information-set-sampler-v1")
    )
    run(parser.parse_args().directory)
