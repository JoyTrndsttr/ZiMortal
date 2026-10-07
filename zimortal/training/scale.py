"""Resumable numeric shards and multiprocessing for six-figure datasets."""

import argparse
import hashlib
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from zimortal.engine import RuleEngine

from .data import example, generate_dataset, make_puzzle
from .runtime import load_model, save_model, tournament
from .train import batch, evaluate


def generator_signature():
    package = Path(__file__).resolve().parents[1]
    files = sorted((package / "engine").glob("*.py")) + [
        package / "model" / "encoding.py",
        package / "training" / "data.py",
    ]
    return {
        str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def verify_generator(root):
    path = root / "generator.json"
    signature = generator_signature()
    if path.exists():
        if json.loads(path.read_text()) != signature:
            raise ValueError("generator code changed; use a new data directory")
    elif any(root.glob("*.npz")):
        raise ValueError("cached shards have no generator provenance")
    else:
        path.write_text(json.dumps(signature, indent=2) + "\n")


def write_shard(path, rows):
    offsets = np.cumsum([0] + [len(r[1]) for r in rows], dtype=np.int64)
    np.savez_compressed(
        path,
        features=np.stack([r[0] for r in rows]),
        actions=np.concatenate([r[1] for r in rows]),
        offsets=offsets,
        policy=np.array([r[2] for r in rows], np.int64),
        value=np.array([r[3] for r in rows], np.float32),
        waits=np.stack([r[4] for r in rows]),
    )


def read_shard(path):
    with np.load(path, allow_pickle=False) as z:
        x, a, o, y, v, w = [
            z[k] for k in ("features", "actions", "offsets", "policy", "value", "waits")
        ]
    if len(o) != len(x) + 1 or o[0] != 0 or o[-1] != len(a) or (np.diff(o) <= 0).any():
        raise ValueError("invalid action offsets")
    if ((y < 0) | (y >= np.diff(o))).any():
        raise ValueError("illegal teacher index")
    return [(x[i], a[o[i] : o[i + 1]], int(y[i]), float(v[i]), w[i]) for i in range(len(x))]


def shard_job(job):
    directory, kind, index, namespace, puzzles, games = job
    path = Path(directory) / f"{kind}-{index:04d}.npz"
    meta = path.with_suffix(".json")
    config = {"kind": kind, "namespace": namespace, "puzzles": puzzles, "games": games}
    if meta.exists() and path.exists():
        saved = json.loads(meta.read_text())
        if (
            saved["config"] != config
            or saved["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
        ):
            raise ValueError("shard configuration/hash mismatch")
        return saved
    start = time.time()
    rows, strata = generate_dataset(puzzles, games, namespace)
    write_shard(path, rows)
    saved = {
        "file": path.name,
        "config": config,
        "samples": len(rows),
        "strata": strata,
        "seconds": time.time() - start,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    meta.write_text(json.dumps(saved, indent=2) + "\n")
    return saved


def build(directory, workers=4):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    verify_generator(root)
    # A namespace is a complete 100000-seed block. Never share a block across splits.
    jobs = [(str(root), "train-puzzle", i, 400 + i, 1000, 0) for i in range(50)]
    jobs += [(str(root), "train-game", i, 500 + i, 0, 32) for i in range(64)]
    jobs += [(str(root), "valid-puzzle", i, 800 + i, 1000, 0) for i in range(2)]
    jobs += [(str(root), "valid-game", i, 850 + i, 0, 32) for i in range(3)]
    manifest = {
        "version": 1,
        "value_target": "immediate hu or own-draw readiness",
        "teacher": "visible exact waits plus cohesion",
        "shards": [],
    }
    with ProcessPoolExecutor(max_workers=workers) as pool:
        # Ordered completion makes the manifest deterministic apart from measured timing.
        for record in pool.map(shard_job, jobs):
            manifest["shards"].append(record)
            print(
                json.dumps(
                    {
                        "completed": len(manifest["shards"]),
                        "shards": len(jobs),
                        "file": record["file"],
                        "samples": record["samples"],
                    }
                ),
                flush=True,
            )
    manifest["train_samples"] = sum(
        s["samples"] for s in manifest["shards"] if s["config"]["kind"].startswith("train")
    )
    manifest["validation_samples"] = sum(
        s["samples"] for s in manifest["shards"] if s["config"]["kind"].startswith("valid")
    )
    if manifest["train_samples"] < 100000:
        raise RuntimeError("not enough actual decision samples; extend game shards")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def deduplicate(training, valid):
    """Remove repeated encoded decisions and exact validation overlap."""

    def key(row):
        return hashlib.sha256(row[0].tobytes() + row[1].tobytes()).digest()

    seen = set()
    unique = []
    for row in training:
        fingerprint = key(row)
        if fingerprint not in seen:
            unique.append(row)
            seen.add(fingerprint)
    validation_seen = set()
    holdout = []
    for row in valid:
        fingerprint = key(row)
        if fingerprint not in seen and fingerprint not in validation_seen:
            holdout.append(row)
            validation_seen.add(fingerprint)
    return unique, holdout


def train(args):
    torch.set_num_threads(2)
    torch.manual_seed(44)
    root = Path(args.data)
    verify_generator(root)
    manifest = json.loads((root / "manifest.json").read_text())
    training = []
    valid = []
    for record in manifest["shards"]:
        path = root / record["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("corrupt dataset shard")
        rows = read_shard(path)
        if len(rows) != record["samples"]:
            raise ValueError("shard sample count mismatch")
        (training if record["config"]["kind"].startswith("train") else valid).extend(rows)
    if len(training) < 100000 or len(training) != manifest["train_samples"]:
        raise ValueError("six-figure dataset required")
    raw_train, raw_valid = len(training), len(valid)
    training, valid = deduplicate(training, valid)
    if len(training) < 100000 or not valid:
        raise ValueError(
            "at least 100000 unique training inputs and independent validation required"
        )
    model = load_model(args.resume)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(44)
    logs = []
    best = float("inf")
    initial = evaluate(model, valid)
    print(json.dumps({"train": len(training), "valid": len(valid), "initial": initial}), flush=True)
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(training))
        loss_sum = 0.0
        start = time.time()
        for offset in range(0, len(order), 128):
            indices = order[offset : offset + 128]
            x, a, m, y, v, w = batch([training[i] for i in indices])
            logits, value, wait = model(x, a, m)
            loss = (
                torch.nn.functional.cross_entropy(logits, y)
                + 0.2 * torch.nn.functional.binary_cross_entropy_with_logits(value, v)
                + 0.1 * torch.nn.functional.binary_cross_entropy_with_logits(wait, w)
            )
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite supervised loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
        metrics = {
            "epoch": epoch + 1,
            "loss": loss_sum / len(training),
            "seconds": time.time() - start,
            **evaluate(model, valid),
        }
        logs.append(metrics)
        if metrics["policy_cross_entropy"] < best:
            best = metrics["policy_cross_entropy"]
            save_model(
                model,
                args.output,
                training_samples=len(training),
                value_target=manifest["value_target"],
                epoch=epoch + 1,
                dataset_sha256=hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
            )
        print(json.dumps(metrics), flush=True)
    model = load_model(args.output)
    report = {
        "raw_training_samples": raw_train,
        "raw_validation_samples": raw_valid,
        "training_samples": len(training),
        "validation_samples": len(valid),
        "exact_input_overlap_after_filtering": 0,
        "data": str(root),
        "generator_signature": generator_signature(),
        "dataset_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        "resume": args.resume,
        "seed": 44,
        "device": "cpu",
        "torch": torch.__version__,
        "initial_validation": initial,
        "epochs": logs,
        "selected_by": "lowest validation policy cross entropy",
        "checkpoint": args.output,
        "sha256": hashlib.sha256(Path(args.output).read_bytes()).hexdigest(),
        "random_tournament": tournament(model, list(range(16000, 16050))),
        "teacher_tournament": tournament(model, list(range(17000, 17020)), opponent="teacher"),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["random_tournament"]), flush=True)


def audit(directory, output="docs/training/scale100k-data-audit.json"):
    root = Path(directory)
    verify_generator(root)
    manifest = json.loads((root / "manifest.json").read_text())
    namespaces = [r["config"]["namespace"] for r in manifest["shards"]]
    if len(namespaces) != len(set(namespaces)):
        raise ValueError("seed namespaces overlap")
    engine = RuleEngine()
    decisions = games = steps = puzzles = 0
    game_records = [r for r in manifest["shards"] if r["config"]["kind"] == "train-game"]
    for record in game_records[::8]:
        namespace = record["config"]["namespace"]
        rows = read_shard(root / record["file"])
        rng = random.Random(namespace)
        cursor = 0
        for i in range(3):
            initial = engine.new_game(namespace * 100000 + i)
            state = initial
            while not state.terminal:
                actions = engine.legal_actions(state)
                if len(actions) > 1:
                    expected = example(engine.observation(state, actions[0].player))
                    actual = rows[cursor]
                    for field in (0, 1, 4):
                        if not np.array_equal(expected[field], actual[field]):
                            raise RuntimeError(
                                "saved decision does not match reproducible observation"
                            )
                    if expected[2:4] != actual[2:4]:
                        raise RuntimeError("teacher targets do not match")
                    cursor += 1
                    decisions += 1
                state = engine.step(state, rng.choice(actions))
                state.validate()
                steps += 1
            if engine.replay(initial, state.history).serialize() != state.serialize():
                raise RuntimeError("dataset game replay mismatch")
            games += 1
    for record in [r for r in manifest["shards"] if r["config"]["kind"] == "train-puzzle"][::10]:
        rows = read_shard(root / record["file"])
        for i in range(10):
            expected = example(make_puzzle(record["config"]["namespace"] * 100000 + i)[0])
            if (
                any(not np.array_equal(expected[j], rows[i][j]) for j in (0, 1, 4))
                or expected[2:4] != rows[i][2:4]
            ):
                raise RuntimeError("puzzle regeneration mismatch")
            puzzles += 1
    report = {
        "games": games,
        "decisions": decisions,
        "steps": steps,
        "puzzles": puzzles,
        "seed_namespaces_disjoint": True,
        "saved_inputs_and_labels_verified": True,
        "conservation_and_replay_verified": True,
    }
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def benchmark():
    torch.set_num_threads(2)
    result = {
        "random_seeds": list(range(16000, 16050)),
        "teacher_seeds": list(range(17000, 17020)),
        "evaluation_only": True,
    }
    for name in ("round2", "round3"):
        model = load_model(f"checkpoints/{name}-resnet.pt")
        result[name] = {
            "random_tournament": tournament(model, result["random_seeds"]),
            "teacher_tournament": tournament(model, result["teacher_seeds"], opponent="teacher"),
        }
        print(name, result[name], flush=True)
    result["teacher_vs_random"] = tournament(None, result["random_seeds"], model_policy="teacher")
    result["random_vs_random"] = tournament(None, result["random_seeds"], model_policy="random")
    Path("docs/training/scale100k-baselines.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["build", "train", "audit", "benchmark"])
    p.add_argument("--data", default="data/generated/scale100k")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--resume", default="checkpoints/round2-resnet.pt")
    p.add_argument("--output", default="checkpoints/scale100k-resnet.pt")
    p.add_argument("--report", default="docs/training/scale100k.json")
    args = p.parse_args()
    if args.workers < 1 or args.epochs < 1:
        p.error("workers and epochs must be positive")
    if args.command == "build":
        build(args.data, args.workers)
    elif args.command == "benchmark":
        benchmark()
    elif args.command == "audit":
        audit(args.data)
    else:
        train(args)


if __name__ == "__main__":
    main()
