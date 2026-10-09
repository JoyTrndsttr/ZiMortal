"""Broad mining with separate cheap and deep compute lanes.

The frozen active-v2 producer and statistical configuration remain unchanged.
A copied data version carries every independent particle, root confidence ID,
and certified label; only the allocation of future compute changes.
"""

import argparse
import json
import math
import multiprocessing
import os
import shutil
import sqlite3
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import torch

from . import active as core
from .rollout_engine import initialize_worker


def verify_producer(configuration):
    for path, expected in configuration["sources"].items():
        if core.digest(path) != expected:
            raise ValueError(f"frozen producer changed: {path}")
    for name in ("parent", "current", "historical"):
        if core.digest(configuration[f"{name}_path"]) != configuration[f"{name}_sha256"]:
            raise ValueError(f"frozen checkpoint changed: {name}")


def copy_evidence(source, target):
    """Snapshot a stopped queue; hashes and raw evidence are not rewritten."""
    source, target = Path(source), Path(target)
    if target.exists():
        raise ValueError("copy target already exists")
    configuration = json.loads((source / "config.json").read_text())
    verify_producer(configuration)
    old_status = json.loads((source / "progress.json").read_text())
    pid = old_status.get("process_id")
    if pid:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError("stop the source coordinator before copying")
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns(
            "queue.sqlite*", "*.tmp", "scheduler.json", "scheduler-policy-history.jsonl"
        ),
    )
    with (
        sqlite3.connect(source / "queue.sqlite") as old,
        sqlite3.connect(target / "queue.sqlite") as new,
    ):
        old.backup(new)
        for key, path in new.execute("SELECT input_hash,path FROM roots").fetchall():
            new.execute(
                "UPDATE roots SET path=? WHERE input_hash=?",
                (str(target / "candidates" / Path(path).name), key),
            )
        new.commit()
    core.atomic_json(
        target / "lineage.json",
        {
            "copied_from": str(source.resolve()),
            "producer_config_sha256": core.digest(source / "config.json"),
            "raw_evidence_unchanged": True,
            "confidence_indices_preserved": True,
            "source_progress": old_status,
            "source_scheduler": json.loads((source / "scheduler.json").read_text())
            if (source / "scheduler.json").exists()
            else None,
            "source_lineage": json.loads((source / "lineage.json").read_text())
            if (source / "lineage.json").exists()
            else None,
        },
    )


_COST_CACHE = {}


def priority(row, current, configuration):
    _, _, _, score, n, labels, actions, split = row
    deficits = [
        max(0, 1 - current["training_strata"].get(t, 0) / max(1, configuration["quotas"].get(t, 0)))
        for t in json.loads(labels)
        if t in configuration["quotas"]
    ]
    validation_bonus = (
        4
        if split == "valid" and current["qualified"].get("valid", 0) < configuration["validation"]
        else 0
    )
    if row[2] == "deferred":
        path = Path(row[1]).parent.parent / "labels" / (row[0] + ".npz")
        stamp = path.stat().st_mtime_ns
        cached = _COST_CACHE.get(row[0])
        if cached is None or cached[0] != stamp:
            with np.load(path, allow_pickle=False) as data:
                q = data["q"].astype(float) * 100
                best = int(q.argmax())
                intervals = data["intervals"][best]
                gaps = q[best] - q
                other = np.arange(len(q)) != best
                radius = np.maximum(gaps - intervals[:, 0], intervals[:, 1] - gaps)
                tolerance = np.maximum(1.0, gaps)
                ratio = float(np.max(radius[other] / tolerance[other]))
                projected = n * max(2.0, ratio, ratio * ratio)
                stable_bonus = (
                    8
                    if json.loads(path.with_suffix(".json").read_text())["trace"][-1]["resolved"]
                    else 0
                )
                cost = math.log2(max(1, projected - n) * max(1, actions))
            cached = (stamp, cost, stable_bonus)
            _COST_CACHE[row[0]] = cached
        return score + 8 * max(deficits, default=0) + validation_bonus + cached[2] - cached[1]
    return (
        score
        + 8 * max(deficits, default=0)
        + validation_bonus
        - 2 * math.log2(max(1, n / configuration["adaptive"]["minimum"]))
        - math.log2(max(1, actions))
    )


def choose_job(db, current, configuration, ticket, deep_running, deep_limit):
    """Fresh roots always have capacity even when a deep look takes minutes."""
    preferred = ("screen", "refine", "deferred", "refine")[ticket % 4]
    for status in dict.fromkeys((preferred, "refine", "screen", "deferred")):
        if status == "deferred" and deep_running >= 3:
            continue
        rows = db.execute(
            "SELECT input_hash,path,status,priority,n,tags,actions,split FROM roots WHERE status=? AND (?!='deferred' OR (n<? AND n*4*actions<=?)) ORDER BY priority DESC LIMIT 512",
            (status, status, deep_limit, core.EVIDENCE_BYTES),
        ).fetchall()
        if rows:
            return max(rows, key=lambda row: priority(row, current, configuration))
    return None


def reconcile(db, root, fresh_limit):
    """Recover completed atomic writes before deciding which lane owns a root."""
    for key, status, n in db.execute("SELECT input_hash,status,n FROM roots").fetchall():
        if status == "screen_running":
            db.execute("UPDATE roots SET status='screen' WHERE input_hash=?", (key,))
        elif status in ("refine_running", "deep_running"):
            db.execute("UPDATE roots SET status='refine' WHERE input_hash=?", (key,))
        path = root / "labels" / f"{key}.json"
        if path.exists() and status not in ("sampling_failed", "rule_unresolved", "memory_limited"):
            meta = json.loads(path.read_text())
            if core.digest(root / meta["file"]) != meta["sha256"]:
                raise ValueError("saved particle evidence changed")
            n = meta["rollouts_per_action"]
            status = (
                "qualified" if meta["qualified"] else "deferred" if n >= fresh_limit else "refine"
            )
            db.execute("UPDATE roots SET status=?,n=? WHERE input_hash=?", (status, n, key))
    db.commit()


def mine(db, root, configuration, models):
    index = db.execute("SELECT game FROM cursor WHERE id=1").fetchone()[0]
    if index >= configuration["max_games"]:
        return False
    seed = configuration["start_seed"] + index
    try:
        candidates = core.mine_game(seed, models)
    except core.RuleClarificationRequired as exc:
        core.atomic_json(
            root / "questions" / f"game-{seed}.json",
            {
                "seed": seed,
                "message": str(exc),
                "context": getattr(exc, "rollout_context", None),
                "used_for_training": False,
            },
        )
        candidates = []
    for meta in candidates:
        key = meta["input_hash"]
        if db.execute("SELECT 1 FROM roots WHERE input_hash=?", (key,)).fetchone():
            continue
        path = root / "candidates" / f"{key}.json"
        core.atomic_json(path, meta)
        db.execute(
            "INSERT INTO roots(input_hash,split,path,status,priority,tags,actions) VALUES(?,?,?,?,?,?,?)",
            (
                key,
                "valid" if seed % 10 == 9 else "train",
                str(path),
                "screen",
                meta["priority"],
                json.dumps(meta["tags"]),
                len(meta["observation"]["legal_actions"]),
            ),
        )
    db.execute("UPDATE cursor SET game=? WHERE id=1", (index + 1,))
    db.commit()
    return True


def allocate_confidence(db, root, key, configuration, fresh_limit):
    index = db.execute("SELECT confidence_index FROM roots WHERE input_hash=?", (key,)).fetchone()[
        0
    ]
    if index is None:
        index = db.execute("SELECT game FROM cursor WHERE id=2").fetchone()[0] + 1
        db.execute("UPDATE cursor SET game=? WHERE id=2", (index,))
        db.execute("UPDATE roots SET confidence_index=? WHERE input_hash=?", (index, key))
        db.commit()  # Allocation is durable before any certified sample.
    path = root / "candidates" / f"{key}.json"
    meta = json.loads(path.read_text())
    meta.update(
        confidence_index=index,
        root_alpha=configuration["adaptive"]["alpha"] / (index * (index + 1)),
        particle_limit=fresh_limit,
    )
    core.atomic_json(path, meta)


def postprocess(root, configuration, schedule, train_after):
    manifest = core.finish_manifest(sqlite3.connect(root / "queue.sqlite"), root, configuration)
    manifest["scheduler"] = schedule
    core.atomic_json(root / "manifest.json", manifest)
    core.audit_dataset(root)
    status = json.loads((root / "progress.json").read_text())
    status.update(status="ready")
    core.atomic_json(root / "progress.json", status)
    if not train_after:
        return
    python = str(Path(".venv-cuda/bin/python").resolve())
    parent, output = configuration["parent_path"], "checkpoints/activeq-resnet.pt"
    status.update(status="training", training_started=True)
    core.atomic_json(root / "progress.json", status)
    subprocess.run(
        [
            python,
            "-m",
            "zimortal.training.cashq",
            "train",
            "--data",
            str(root),
            "--parent",
            parent,
            "--output",
            output,
            "--report",
            str(root / "training.json"),
            "--epochs",
            "60",
            "--device",
            "cuda",
            "--selection",
            "policy_regret",
        ]
        + (["--continue-training"] if Path(output + ".recovery.pt").exists() else []),
        check=True,
    )
    subprocess.run(
        [
            python,
            "-m",
            "zimortal.training.cashq",
            "evaluate",
            "--data",
            str(root),
            "--parent",
            parent,
            "--output",
            output,
            "--report",
            str(root / "comparison.json"),
            "--dev-start",
            "3000000",
            "--eval-start",
            "3001000",
            "--eval-seeds",
            "200",
        ],
        check=True,
    )
    subprocess.run(
        [
            python,
            "-m",
            "zimortal.training.review",
            "--checkpoint",
            "checkpoints/activeq-resnet.gated.pt",
            "--start",
            "3002000",
            "--games",
            "100",
            "--teacher",
            "huxi",
            "--output",
            str(root / "game-review.json"),
        ],
        check=True,
    )
    status.update(status="complete", report=str(root / "comparison.json"))
    core.atomic_json(root / "progress.json", status)


def run(args):
    root = Path(args.data)
    if not root.exists():
        copy_evidence(args.from_data, root)
    configuration = json.loads((root / "config.json").read_text())
    verify_producer(configuration)
    schedule = {
        "source_sha256": core.digest(__file__),
        "source": __file__,
        "fresh_limit": args.fresh_limit,
        "deep_limit": args.deep_limit,
        "candidate_low_water": args.candidate_low_water,
        "mine_every_seconds": args.mine_every_seconds,
        "max_deep_workers": 3,
        "workers": args.workers,
        "rollout_engine_sha256": core.digest("zimortal/training/rollout_engine.py"),
    }
    previous = root / "scheduler.json"
    if (
        previous.exists()
        and json.loads(previous.read_text())["source_sha256"] != schedule["source_sha256"]
    ):
        raise ValueError("scheduler source changed; create a new copied data version")
    if previous.exists() and json.loads(previous.read_text()).get("rollout_engine_sha256") != schedule["rollout_engine_sha256"]:
        raise ValueError("frozen rollout optimization changed; create a new data version")
    if previous.exists() and json.loads(previous.read_text()) != schedule:
        with (root / "scheduler-policy-history.jsonl").open("a") as file:
            file.write(json.dumps(json.loads(previous.read_text())) + "\n")
    core.atomic_json(previous, schedule)
    db = sqlite3.connect(root / "queue.sqlite")
    db.execute("PRAGMA journal_mode=WAL")
    reconcile(db, root, args.fresh_limit)
    torch.set_num_threads(1)
    models = [
        core.load_model(configuration[f"{name}_path"])
        for name in ("parent", "current", "historical")
    ]
    if any(
        not torch.equal(value, models[1].parent.state_dict()[key])
        for key, value in models[0].state_dict().items()
    ):
        raise ValueError("frozen parent mismatch")
    futures, ticket, completed = {}, 0, 0
    last_mine, started = 0, time.monotonic()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize_worker,
        initargs=(configuration["parent_path"],),
    ) as pool:
        while True:
            current = core.progress(db, configuration)
            current.update(
                scheduler=schedule,
                completed_tasks=completed,
                elapsed_seconds=time.monotonic() - started,
            )
            core.atomic_json(root / "progress.json", current)
            ready = (
                current["qualified"].get("train", 0) >= configuration["target"]
                and current["qualified"].get("valid", 0) >= configuration["validation"]
                and all(
                    current["training_strata"].get(k, 0) >= v
                    for k, v in configuration["quotas"].items()
                )
            )
            smoke_done = args.smoke_tasks and completed >= args.smoke_tasks
            if (ready or smoke_done) and not futures:
                if smoke_done:
                    current.update(status="smoke_complete")
                    core.atomic_json(root / "progress.json", current)
                    return
                break
            if not ready and not smoke_done:
                light = db.execute(
                    "SELECT COUNT(*) FROM roots WHERE status IN ('screen','refine','screen_running','refine_running')"
                ).fetchone()[0]
                if (
                    light < args.candidate_low_water
                    and time.monotonic() - last_mine >= args.mine_every_seconds
                ):
                    mine(db, root, configuration, models)
                    last_mine = time.monotonic()
                while len(futures) < args.workers:
                    deep_running = sum(deep for _, _, deep in futures.values())
                    row = choose_job(
                        db,
                        current,
                        configuration,
                        ticket,
                        deep_running,
                        min(args.deep_limit, configuration["adaptive"]["maximum"]),
                    )
                    if row is None:
                        break
                    key, path, status, _, n, _, actions, _ = row
                    deep = status == "deferred"
                    mode = "screen" if status == "screen" else "refine"
                    if mode == "refine":
                        meta = json.loads(Path(path).read_text())
                        cap = min(
                            args.deep_limit if deep else args.fresh_limit,
                            configuration["adaptive"]["maximum"],
                            2 ** int(math.log2(core.EVIDENCE_BYTES // (2 * actions))),
                        )
                        meta["particle_limit"] = max(
                            2 * configuration["adaptive"]["minimum"],
                            min(cap, 2 * n) if deep else cap,
                        )
                        core.atomic_json(path, meta)
                    future = pool.submit(core.process_root, (str(root), path, mode, configuration))
                    futures[future] = (key, mode, deep)
                    db.execute(
                        "UPDATE roots SET status=? WHERE input_hash=?",
                        ("deep_running" if deep else mode + "_running", key),
                    )
                    db.commit()
                    ticket += 1
            if not futures:
                if (
                    db.execute("SELECT game FROM cursor WHERE id=1").fetchone()[0]
                    >= configuration["max_games"]
                ):
                    raise RuntimeError(
                        "mining and staged budgets exhausted without enough certified roots; no training"
                    )
                time.sleep(0.05)
                continue
            done, _ = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                key, mode, deep = futures.pop(future)
                try:
                    result = future.result()
                    if mode == "screen":
                        allocate_confidence(db, root, key, configuration, args.fresh_limit)
                    status = result["status"]
                    if (
                        status != "qualified"
                        and mode == "refine"
                        and (deep or result["n"] >= args.fresh_limit)
                    ):
                        status = "deferred"
                    db.execute(
                        "UPDATE roots SET status=?,priority=?,n=? WHERE input_hash=?",
                        (status, result["priority"], result["n"], key),
                    )
                except core.SamplingFailure as exc:
                    db.execute(
                        "UPDATE roots SET status='sampling_failed' WHERE input_hash=?", (key,)
                    )
                    print(json.dumps({"sampling_failed": key, "message": str(exc)}), flush=True)
                except core.RuleClarificationRequired as exc:
                    db.execute(
                        "UPDATE roots SET status='rule_unresolved' WHERE input_hash=?", (key,)
                    )
                    core.atomic_json(
                        root / "questions" / f"{key}.json",
                        {
                            "input_hash": key,
                            "mode": mode,
                            "message": str(exc),
                            "context": getattr(exc, "rollout_context", None),
                            "used_for_training": False,
                        },
                    )
                db.commit()
                completed += 1
                print(json.dumps(core.progress(db, configuration), ensure_ascii=False), flush=True)
    verify_producer(configuration)
    postprocess(root, configuration, schedule, args.train_after)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/generated/active-v5")
    p.add_argument("--from-data", default="data/generated/active-v4")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--fresh-limit", type=int, default=1024)
    p.add_argument("--deep-limit", type=int, default=262144)
    p.add_argument("--candidate-low-water", type=int, default=256)
    p.add_argument("--mine-every-seconds", type=float, default=2)
    p.add_argument("--smoke-tasks", type=int, default=0)
    p.add_argument("--train-after", action="store_true")
    args = p.parse_args()
    if (
        args.workers < 1
        or args.candidate_low_water < 1
        or args.mine_every_seconds <= 0
        or args.smoke_tasks < 0
    ):
        p.error("invalid scheduling limits")
    if (
        args.fresh_limit < 256
        or args.deep_limit < args.fresh_limit
        or any(v & (v - 1) for v in (args.fresh_limit, args.deep_limit))
    ):
        p.error("particle limits must be powers of two with 256 <= fresh <= deep")
    if args.smoke_tasks and args.train_after:
        p.error("smoke cannot train")
    try:
        run(args)
    except Exception as exc:
        root = Path(args.data)
        if root.exists():
            path = root / "progress.json"
            status = json.loads(path.read_text()) if path.exists() else {}
            status.update(status="failed", error=str(exc), timestamp=time.time())
            core.atomic_json(path, status)
        raise


if __name__ == "__main__":
    main()
