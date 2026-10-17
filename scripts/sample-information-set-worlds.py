"""Lock outcome-blind public roots, then sample; never continue games past root."""

import argparse
import gzip
import hashlib
import json
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from zimortal.training.information_set_sampler import (
    InformationSetSampler,
    canonical,
    public_from_json,
)

SEED = 20261019
QUOTAS = {"discard": 20, "chi/pass": 10, "peng/pass": 10, "hu/pass": 10}
SOURCE = Path("data/generated/counterfactual-ranking-v1")
DEFAULT = Path("data/generated/information-set-sampler-v1")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def seeded(root_id, index):
    return int.from_bytes(
        hashlib.sha256(canonical([SEED, root_id, index]).encode()).digest()[:8], "little"
    )


def frozen_sources():
    paths = list(Path("zimortal/engine").glob("*.py"))
    paths += list(Path("zimortal/belief").glob("*.py"))
    paths += [
        Path("zimortal/training/information_set_sampler.py"),
        Path("scripts/sample-information-set-worlds.py"),
        Path("scripts/audit-information-set-sampler.py"),
    ]
    return {str(p): digest(p) for p in sorted(paths)}


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "public").mkdir()
    (directory / "worlds").mkdir()
    manifest = json.loads((SOURCE / "locked-state-manifest.json").read_text())
    rng = random.Random(SEED)
    chosen = []
    for family, quota in QUOTAS.items():
        eligible = sorted((r for r in manifest if r["family"] == family), key=lambda r: r["id"])
        chosen.extend(rng.sample(eligible, quota))
    records = []
    for row in chosen:
        source = SOURCE / row["file"]
        assert digest(source) == row["sha256"]
        # Deliberately project immediately: no root/initial/suffix/prediction is retained.
        with gzip.open(source, "rt") as file:
            observation = json.load(file)["observation"]
        public = public_from_json(observation)
        target = directory / "public" / f"{row['id']}.json"
        save(target, observation)
        records.append(
            {
                "id": row["id"],
                "family": row["family"],
                "public_sha256": digest(target),
                "observation_sha256": public.sha256,
                "audit_source": str(source),
                "audit_source_sha256": row["sha256"],
            }
        )
    save(directory / "locked-root-manifest.json", records)
    inputs = json.loads((SOURCE / "input-sha256.json").read_text())
    checkpoints = {p: h for p, h in inputs.items() if p.endswith(".pt")}
    assert all(digest(p) == h for p, h in checkpoints.items())
    save(
        directory / "config.json",
        {
            "experiment": "information-set-sampler-v1",
            "seed": SEED,
            "quotas": QUOTAS,
            "worlds_per_root": 1000,
            "max_proposals_per_world": 200000,
            "workers": 2,
            "distribution": "exchangeable current physical slots conditioned on necessary counts and full legal prefix replay",
            "behavior_likelihood": "OFF",
            "training": False,
            "future_rollouts": 0,
            "root_manifest_sha256": digest(directory / "locked-root-manifest.json"),
            "prior_manifest_sha256": digest(SOURCE / "locked-state-manifest.json"),
            "sources": frozen_sources(),
            "checkpoints": checkpoints,
            "collapse_flag": "empirical effective joint-hand count <10 AND more than 1000 DP count allocations; requires investigation, not automatic proof",
            "toy_seed_count": 512,
        },
    )
    save(
        directory / "lock.json",
        {
            "locked_before_sampling": True,
            "config_sha256": digest(directory / "config.json"),
            "manifest_sha256": digest(directory / "locked-root-manifest.json"),
            "time_unix": time.time(),
        },
    )
    print("LOCKED", len(records), digest(directory / "locked-root-manifest.json"), flush=True)


def sample_root(job):
    # Worker receives only a PUBLIC path, root ID, and fixed sampling config.
    directory, root_id, family, config = job
    directory = Path(directory)
    public = public_from_json(json.loads((directory / "public" / f"{root_id}.json").read_text()))
    sampler = InformationSetSampler(public)
    output = directory / "worlds" / f"{root_id}.jsonl.gz"
    if output.exists():
        raise FileExistsError("no silent overwrite/resampling")
    tmp = output.with_suffix(".tmp")
    total_attempts, rejections, started = 0, Counter(), time.perf_counter()
    with tmp.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as file:
        for i in range(config["worlds_per_root"]):
            seed = seeded(root_id, i)
            begin = time.perf_counter()
            world, attempts, rejects = sampler.sample(seed, config["max_proposals_per_world"])
            elapsed = time.perf_counter() - begin
            total_attempts += attempts
            rejections.update(rejects)
            row = {
                "index": i,
                "seed": seed,
                "world": {"hands": world.hands, "wall": world.wall},
                "attempts": attempts,
                "rejections": rejects,
                "seconds": elapsed,
            }
            file.write((canonical(row) + "\n").encode())
            if i % 100 == 99:
                save(
                    directory / f"{root_id}-progress.json",
                    {
                        "accepted": i + 1,
                        "proposals": total_attempts,
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                )
    tmp.replace(output)
    result = {
        "id": root_id,
        "family": family,
        "accepted": config["worlds_per_root"],
        "proposals": total_attempts,
        "rejections": dict(rejections),
        "seconds": time.perf_counter() - started,
        "sha256": digest(output),
    }
    save(directory / f"{root_id}-sampling.json", result)
    print(
        root_id, family, result["accepted"], total_attempts, round(result["seconds"], 2), flush=True
    )
    return result


def run(directory):
    config = json.loads((directory / "config.json").read_text())
    lock = json.loads((directory / "lock.json").read_text())
    assert digest(directory / "config.json") == lock["config_sha256"]
    assert digest(directory / "locked-root-manifest.json") == lock["manifest_sha256"]
    assert frozen_sources() == config["sources"]
    manifest = json.loads((directory / "locked-root-manifest.json").read_text())
    jobs = []
    for root in manifest:
        public_path = directory / "public" / f"{root['id']}.json"
        assert digest(public_path) == root["public_sha256"]
        jobs.append((str(directory), root["id"], root["family"], config))
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=config["workers"]) as pool:
        results = list(pool.map(sample_root, jobs, chunksize=1))
    save(
        directory / "sampling-result.json",
        {
            "roots": results,
            "wall_seconds": time.perf_counter() - started,
            "training_updates": 0,
            "future_rollouts": 0,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run"])
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else run)(args.directory)


if __name__ == "__main__":
    main()
