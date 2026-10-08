"""Resumable active-state mining, adaptive refinement, and a strict training gate.

The coordinator and workers exchange serialized Observation only. Screening
particles never enter certified labels. All persistence is bounded per root;
training reads compact targets without loading the retained raw MC evidence.
"""

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import random
import shutil
import sqlite3
import subprocess
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from zimortal.belief.conditional import HistorySampler
from zimortal.belief.sampling import SamplingFailure
from zimortal.engine import (
    Action,
    ActionType,
    Meld,
    MeldType,
    Observation,
    PendingTile,
    RuleClarificationRequired,
    RuleEngine,
    SourceType,
    is_red,
)
from zimortal.engine.observation import PublicPlayer
from zimortal.model.encoding import encode_action, encode_observation

from .adaptive import AdaptiveConfig, adaptive_rollout, relations
from .cashq import digest
from .rollout import teacher
from .runtime import choose, load_model
from .valuation import action_scores

VERSION = 2
EVIDENCE_BYTES = 128 * 1024 * 1024
QUOTAS = {"chi_bi": 1000, "peng_pass": 500, "hu_pass": 200, "fan_boundary": 750, "late": 1500}
PARENT = None


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    required = 2 * 1024**3 + sum(np.asarray(value).nbytes for value in arrays.values())
    if shutil.disk_usage(Path(path).parent).free < required:
        raise OSError("insufficient disk reserve for atomic evidence; no training")
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, **arrays)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def payment_statistics(values, reference):
    """Chunked variance without an N×actions float64 temporary."""
    n, actions = values.shape
    sums, squares, pair_sums, pair_squares = [np.zeros(actions) for _ in range(4)]
    for start in range(0, n, 4096):
        block = values[start : start + 4096].astype(np.float64)
        paired = block - block[:, reference, None]
        sums += block.sum(0)
        squares += (block * block).sum(0)
        pair_sums += paired.sum(0)
        pair_squares += (paired * paired).sum(0)
    mean = sums / n
    se = np.sqrt(np.maximum(0, squares - sums * mean) / (n - 1) / n)
    paired_se = np.sqrt(np.maximum(0, pair_squares - pair_sums * pair_sums / n) / (n - 1) / n)
    return mean, se, paired_se


def serialize_observation(obs):
    return json.loads(json.dumps(asdict(obs), default=lambda value: sorted(value)))


def deserialize_observation(data):
    data = dict(data)

    def action(a):
        return Action(
            ActionType(a["kind"]),
            a["player"],
            a["tile"],
            tuple(a["chi"]),
            tuple(tuple(g) for g in a["bi"]),
            a["source_player"],
            SourceType(a["source_type"]) if a["source_type"] else None,
            a["forced"],
        )

    data["players"] = tuple(
        PublicPlayer(
            p["hand_count"], tuple(Meld(MeldType(m["kind"]), tuple(m["tiles"])) for m in p["melds"])
        )
        for p in data["players"]
    )
    for key in ("hand", "river"):
        data[key] = tuple(data[key])
    for key in ("passed_peng", "passed_chi"):
        data[key] = frozenset(data[key])
    for key in ("history", "legal_actions"):
        data[key] = tuple(action(a) for a in data[key])
    if data["pending"]:
        p = data["pending"]
        data["pending"] = PendingTile(p["tile"], p["player"], SourceType(p["source"]))
    return Observation(**data)


def tags(obs):
    kinds = {a.kind for a in obs.legal_actions}
    labels = []
    if ActionType.CHI in kinds:
        labels.append("chi_bi")
    if {ActionType.PENG, ActionType.PASS} <= kinds:
        labels.append("peng_pass")
    if {ActionType.HU, ActionType.PASS} <= kinds:
        labels.append("hu_pass")
    tiles = list(obs.hand) + [t for m in obs.players[obs.player].melds for t in m.tiles]
    red, big = sum(is_red(t) for t in tiles), sum(t >= 10 for t in tiles)
    if (
        red in (0, 1, 2, 3, 4, 5, 9, 10, 11)
        or big in (17, 18, 19)
        or len(tiles) - big in (17, 18, 19)
    ):
        labels.append("fan_boundary")
    if obs.remaining_tiles <= 6:
        labels.append("late")
    return labels


def disagreement(obs, parent, current):
    x = encode_observation(obs, "huxi")
    a = np.stack([encode_action(t, obs.player) for t in obs.legal_actions])
    with torch.inference_mode():
        ensemble, original = current.forward_q(
            torch.from_numpy(x[None]),
            torch.from_numpy(a[None]),
            torch.ones(1, len(a), dtype=torch.bool),
        )
        q = ensemble[0].mean(-1).numpy() * 100
        old = original[0][0].numpy()
    reference = int(old.argmax())
    candidate = int(q.argmax())
    rank_disagreement = float(np.mean(q > q[reference]) + np.mean(old > old[candidate]))
    heuristic = int(action_scores(obs).argmax())
    score = (
        4 * (candidate != reference)
        + 2 * (heuristic != reference)
        + 3 * rank_disagreement
        + min(3, float(np.std(q)) / 10)
    )
    return x, a, reference, candidate, float(score)


def mine_game(seed, models):
    engine = RuleEngine()
    initial = engine.new_game(seed, seed % 3)
    state, rng = initial, random.Random(seed)
    candidates = []
    behavior = seed % 4
    for step in range(1, 1001):
        if state.terminal:
            break
        try:
            acts = engine.legal_actions(state)
        except RuleClarificationRequired as exc:
            exc.rollout_context = {"seed": seed, "step": step, "state": state.serialize()}
            raise
        obs = engine.observation(state, acts[0].player)
        label = tags(obs) if len(acts) > 1 else []
        current_choice = None
        # Empty-deck zero-reward cleanup cannot pad the requested corpus.
        if label and (obs.remaining_tiles or "hu_pass" in label):
            x, a, ref, qbest, score = disagreement(obs, models[0], models[1])
            current_choice = qbest
            key = hashlib.sha256(x.tobytes() + a.tobytes()).hexdigest()
            candidates.append(
                {
                    "input_hash": key,
                    "seed": seed,
                    "dealer": seed % 3,
                    "step": step,
                    "player": obs.player,
                    "stage": "late" if "late" in label else "middle" if step > 15 else "early",
                    "behavior": behavior,
                    "reference": ref,
                    "q_choice": qbest,
                    "tags": label,
                    "priority": score + 2 * ("late" in label) + 2 * ("hu_pass" in label),
                    "observation": serialize_observation(obs),
                }
            )
        if behavior == 1 and len(acts) > 1:
            if current_choice is None:
                current_choice = disagreement(obs, models[0], models[1])[3]
            selected = acts[current_choice]
        else:
            model = models[2] if behavior == 2 else models[0]
            selected = choose(obs, rng, model, policy="teacher" if behavior == 3 else "model")
        state = engine.step(state, selected)
        state.validate()
    else:
        raise RuntimeError("mining game exceeded step limit")
    for candidate in candidates:
        candidate["mining_priority"] = candidate["priority"]
    if engine.replay(initial, state.history).serialize() != state.serialize():
        raise RuntimeError("mining game replay mismatch")
    # Preserve the best candidate in EVERY requested stratum before the global
    # cutoff, otherwise early generic fan boundaries displace rare HU/late roots.
    selected = {
        r["input_hash"]: r
        for label in QUOTAS
        for r in sorted(
            (r for r in candidates if label in r["tags"]), key=lambda r: r["priority"], reverse=True
        )[:2]
    }
    for r in sorted(candidates, key=lambda r: r["priority"], reverse=True)[:4]:
        selected[r["input_hash"]] = r
    return list(selected.values())


def search_priority(meta, result):
    """Screening heuristic only; argmax index on a tie is not disagreement."""
    q = np.asarray(result.q_cash)
    best, reference, candidate = int(q.argmax()), meta["reference"], meta["q_choice"]
    raw = np.asarray(result.outcomes_cash, np.float64)
    difference = raw[:, best, None] - raw
    error = difference.std(axis=0, ddof=1) / math.sqrt(len(raw))
    margin = q[best] - q
    # This only allocates compute, never certifies a training label.
    clear = margin > np.maximum(1, 2 * error)
    priority = (
        meta["mining_priority"]
        + 5 * bool(clear[reference])
        + 3 * bool(clear[candidate])
        + min(4, max(0, margin[reference] - 2 * error[reference]) / max(1, error[reference]))
    )
    return best, float(priority)


def initialize_worker(parent):
    global PARENT
    torch.set_num_threads(1)
    PARENT = load_model(parent)


def process_root(task):
    directory, candidate_path, mode, configuration = task
    root = Path(directory)
    meta = json.loads(Path(candidate_path).read_text())
    obs = deserialize_observation(meta["observation"])
    key = meta["input_hash"]
    reference = meta["reference"]
    # Disjoint streams for low-budget screening and certified refinement.
    seed = int(key[:15], 16) * 2
    if mode == "screen":
        sampler = HistorySampler(obs)
        result = teacher(
            obs,
            model=PARENT,
            rollouts=32,
            seed=seed + 1,
            world_sampler=sampler,
            batched=True,
            reference_index=reference,
            keep_outcomes=True,
            max_attempts=configuration["max_attempts"],
        )
        best, priority = search_priority(meta, result)
        meta.update(
            pilot_q_cash=list(result.q_cash),
            pilot_paired_se_cash=list(result.paired_standard_errors),
            pilot_best=best,
            priority=float(priority),
            pilot_not_training_data=True,
        )
        atomic_json(candidate_path, meta)
        return {"key": key, "status": "refine", "priority": priority, "n": 0}
    path = root / "labels" / f"{key}.npz"
    previous = None
    configuration_hash = hashlib.sha256(
        json.dumps(configuration, sort_keys=True).encode()
    ).hexdigest()
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if (
                str(saved["input_hash"]) != key
                or str(saved["configuration_hash"]) != configuration_hash
            ):
                raise ValueError("resume root/configuration changed")
            previous = saved["outcomes"]
            if str(saved["outcomes_hash"]) != hashlib.sha256(previous.tobytes()).hexdigest():
                raise ValueError("resume particle evidence checksum mismatch")
    config = AdaptiveConfig(
        **{
            **configuration["adaptive"],
            "alpha": meta["root_alpha"],
            "maximum": meta["particle_limit"],
        }
    )

    def checkpoint(values, result, trace, qualified):
        mean, se, paired = payment_statistics(values, reference)
        optimal = np.array([i in result["optimal"] for i in range(len(obs.legal_actions))])
        # Tied top actions receive equal mass; no noisy MC argmax tie-break.
        soft = np.zeros(len(mean), np.float32)
        soft[optimal] = 1 / optimal.sum()
        atomic_npz(
            path,
            x=encode_observation(obs, "huxi"),
            actions=np.stack([encode_action(a, obs.player) for a in obs.legal_actions]),
            q=mean.astype(np.float32) / 100,
            se=se.astype(np.float32) / 100,
            paired_se=paired.astype(np.float32) / 100,
            soft=soft,
            reference=reference,
            outcomes=values,
            intervals=result["intervals"],
            relations=result["relations"],
            optimal=optimal,
            input_hash=key,
            configuration_hash=configuration_hash,
            outcomes_hash=hashlib.sha256(values.tobytes()).hexdigest(),
        )
        evidence = {k: v for k, v in meta.items() if k != "observation"}
        evidence.update(
            file=f"labels/{key}.npz",
            sha256=digest(path),
            rollouts_per_action=len(values),
            actions=len(mean),
            qualified=qualified,
            confidence_method="alpha-spent simultaneous empirical Bernstein",
            tie_cash=config.tie_cash,
            alpha=config.alpha,
            global_alpha=configuration["adaptive"]["alpha"],
            scope=config.scope,
            trace=trace,
            payoff_bounds_cash=result["payoff_bounds_cash"],
            pair_relations_sha256=hashlib.sha256(result["relations"].tobytes()).hexdigest(),
            continuation="frozen huxi",
            independent_conditional_particles=True,
        )
        atomic_json(path.with_suffix(".json"), evidence)

    values, _result, _trace, status = adaptive_rollout(
        obs,
        PARENT,
        config,
        seed=seed,
        reference=reference,
        initial=previous,
        checkpoint=checkpoint,
        max_attempts=configuration["max_attempts"],
        new_look_limit=1,
    )
    return {
        "key": key,
        "status": status if status != "pending" else "refine",
        "priority": meta["priority"],
        "n": len(values),
    }


def progress(db, configuration):
    statuses = dict(db.execute("SELECT status,COUNT(*) FROM roots GROUP BY status"))
    splits = Counter()
    strata = Counter()
    particles = terminals = 0
    for split, labels, actions, n in db.execute(
        "SELECT split,tags,actions,n FROM roots WHERE status='qualified'"
    ):
        splits[split] += 1
        if split == "train":
            strata.update(json.loads(labels))
        particles += n
        terminals += n * actions
    return {
        "status": "generating",
        "target_training_roots": configuration["target"],
        "target_validation_roots": configuration["validation"],
        "qualified": dict(splits),
        "training_strata": dict(strata),
        "quotas": configuration["quotas"],
        "queue": statuses,
        "qualified_particles": particles,
        "qualified_terminal_rollouts": terminals,
        "confidence_method": "alpha-spent simultaneous empirical Bernstein",
        "tie_cash": configuration["adaptive"]["tie_cash"],
        "training_started": False,
        "timestamp": time.time(),
        "process_id": os.getpid(),
        "mined_games": db.execute("SELECT game FROM cursor WHERE id=1").fetchone()[0],
        "refinement_particles": db.execute("SELECT COALESCE(SUM(n),0) FROM roots").fetchone()[0],
        "refinement_terminal_rollouts": db.execute(
            "SELECT COALESCE(SUM(n*actions),0) FROM roots"
        ).fetchone()[0],
    }


def training_gate(manifest, target=5000, validation=500):
    if target < 5000:
        raise ValueError("training requires at least 5000 certified training roots")
    records = manifest["records"]
    if len(records["train"]) < target or len(records["valid"]) < validation:
        raise ValueError("not enough qualified roots; training is forbidden")
    keys = [{r["input_hash"] for r in records[s]} for s in ("train", "valid")]
    seeds = [{r["seed"] for r in records[s]} for s in ("train", "valid")]
    if (
        keys[0] & keys[1]
        or seeds[0] & seeds[1]
        or any(len(keys[i]) != len(records[s]) for i, s in enumerate(("train", "valid")))
    ):
        raise ValueError("duplicate input or shared deal across training and validation")
    if any(
        not r.get("qualified")
        or r.get("confidence_method") != "alpha-spent simultaneous empirical Bernstein"
        for rows in records.values()
        for r in rows
    ):
        raise ValueError("uncertified or approximate labels cannot pass the training gate")
    indices = [r["confidence_index"] for rows in records.values() for r in rows]
    if len(set(indices)) != len(indices) or min(indices) < 1:
        raise ValueError("duplicate adaptive confidence allocation")
    adaptive = manifest["config"]["adaptive"]
    if validation < 500:
        raise ValueError("training requires 500 independent validation roots")
    if any(
        r["rollouts_per_action"] < 2 * adaptive["minimum"] or r["scope"] != adaptive["scope"]
        for rows in records.values()
        for r in rows
    ):
        raise ValueError("incomplete or inconsistent adaptive certification")
    global_alpha = adaptive["alpha"]
    if any(
        not math.isclose(
            r["alpha"], global_alpha / (r["confidence_index"] * (r["confidence_index"] + 1))
        )
        for rows in records.values()
        for r in rows
    ):
        raise ValueError("invalid corpus-wide confidence budget")
    counts = Counter(t for r in records["train"] for t in r["tags"])
    if any(counts[t] < n for t, n in manifest["config"]["quotas"].items()):
        raise ValueError("priority strata quotas not met")
    return counts


def finish_manifest(db, root, configuration):
    records = {"train": [], "valid": []}
    for split, candidate in db.execute(
        "SELECT split,path FROM roots WHERE status='qualified' ORDER BY input_hash"
    ):
        key = json.loads(Path(candidate).read_text())["input_hash"]
        meta = json.loads((root / "labels" / f"{key}.json").read_text())
        if digest(root / meta["file"]) != meta["sha256"]:
            raise ValueError("qualified shard changed")
        records[split].append(meta)
    manifest = {
        "pipeline": "active",
        "config": configuration,
        "records": records,
        "human_records": 0,
        "screening_particles_not_labels": True,
        "semantics": "net cash/100; uniform full-history-consistent particles; frozen huxi continuation",
    }
    training_gate(manifest, configuration["target"], configuration["validation"])
    atomic_json(root / "manifest.json", manifest)
    return manifest


def audit_dataset(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if any(digest(p) != expected for p, expected in manifest["config"]["sources"].items()):
        raise ValueError("pipeline source changed before dataset audit")
    if any(
        digest(manifest["config"][f"{name}_path"]) != manifest["config"][f"{name}_sha256"]
        for name in ("parent", "current", "historical")
    ):
        raise ValueError("pipeline checkpoints changed before dataset audit")
    training_gate(manifest, manifest["config"]["target"], manifest["config"]["validation"])
    checked = Counter()
    for split, rows in manifest["records"].items():
        for index, meta in enumerate(rows):
            path = root / meta["file"]
            if digest(path) != meta["sha256"]:
                raise ValueError("audit shard checksum mismatch")
            obs = deserialize_observation(
                json.loads((root / "candidates" / f"{meta['input_hash']}.json").read_text())[
                    "observation"
                ]
            )
            with np.load(path, allow_pickle=False) as data:
                np.testing.assert_array_equal(encode_observation(obs, "huxi"), data["x"])
                np.testing.assert_array_equal(
                    np.stack([encode_action(a, obs.player) for a in obs.legal_actions]),
                    data["actions"],
                )
                codes, optimal = data["relations"], np.flatnonzero(data["optimal"])
                if (
                    not len(optimal)
                    or not np.isclose(data["soft"].sum(), 1)
                    or any(codes[i, j] != 2 for i in optimal for j in optimal)
                    or any(
                        codes[i, j] != 1
                        for i in optimal
                        for j in range(len(codes))
                        if j not in optimal
                    )
                    or (meta["scope"] == "full_order" and np.any(codes == 0))
                ):
                    raise ValueError("unresolved ranking or invalid policy target")
                if index % max(1, len(rows) // 8) == 0:
                    raw = data["outcomes"]
                    config = AdaptiveConfig(
                        **{**manifest["config"]["adaptive"], "alpha": meta["alpha"]}
                    )
                    look = int(math.log2(len(raw) // config.minimum))
                    recomputed = relations(raw, obs, config, look)
                    previous = relations(raw[: len(raw) // 2], obs, config, look - 1)
                    np.testing.assert_array_equal(recomputed["relations"], data["relations"])
                    np.testing.assert_array_equal(recomputed["optimal"], optimal)
                    np.testing.assert_allclose(
                        recomputed["intervals"], data["intervals"], rtol=0, atol=1e-10
                    )
                    stable = (
                        np.array_equal(previous["relations"], recomputed["relations"])
                        if config.scope == "full_order"
                        else previous["optimal"] == recomputed["optimal"]
                    )
                    if not previous["resolved"] or not recomputed["resolved"] or not stable:
                        raise ValueError("two-look certification does not reproduce")
                    checked[split] += 1
    report = {
        "passed": True,
        "roots": {s: len(r) for s, r in manifest["records"].items()},
        "intervals_recomputed": dict(checked),
        "all_inputs_reconstructed": True,
        "seed_groups_disjoint": True,
        "manifest_sha256": digest(root / "manifest.json"),
    }
    atomic_json(root / "audit.json", report)
    return report


def run(args):
    root = Path(args.data)
    for child in (root, root / "candidates", root / "labels", root / "questions"):
        child.mkdir(parents=True, exist_ok=True)
    configuration = {
        "version": VERSION,
        "runtime": {
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "generation_device": "cpu",
            "worker_threads": 1,
        },
        "target": args.target,
        "validation": args.validation,
        "quotas": {} if args.smoke else QUOTAS,
        "adaptive": asdict(
            AdaptiveConfig(
                args.minimum, args.maximum, args.chunk, args.alpha, args.tie_cash, args.scope
            )
        ),
        "max_attempts": args.max_attempts,
        "start_seed": args.start_seed,
        "max_games": args.max_games,
        "smoke": args.smoke,
        "parent_sha256": digest(args.parent),
        "current_sha256": digest(args.current),
        "historical_sha256": digest("checkpoints/scale100k-resnet.pt"),
        "parent_path": str(Path(args.parent).resolve()),
        "current_path": str(Path(args.current).resolve()),
        "historical_path": str(Path("checkpoints/scale100k-resnet.pt").resolve()),
        "evidence_bytes_per_root": EVIDENCE_BYTES,
        "sources": {
            str(p): digest(p)
            for package in ("engine", "belief", "model", "training")
            for p in sorted((Path("zimortal") / package).glob("*.py"))
        },
    }
    config_path = root / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError(
            "pipeline configuration/source changed; preserve or version the existing dataset"
        )
    atomic_json(config_path, configuration)
    db = sqlite3.connect(root / "queue.sqlite")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute(
        "CREATE TABLE IF NOT EXISTS roots(input_hash TEXT PRIMARY KEY, split TEXT, path TEXT, status TEXT, priority REAL, n INTEGER DEFAULT 0, tags TEXT, actions INTEGER, confidence_index INTEGER UNIQUE)"
    )
    db.execute("CREATE TABLE IF NOT EXISTS cursor(id INTEGER PRIMARY KEY, game INTEGER)")
    db.execute("INSERT OR IGNORE INTO cursor VALUES(1,0)")
    db.execute("INSERT OR IGNORE INTO cursor VALUES(2,0)")
    db.execute("UPDATE roots SET status='screen' WHERE status='screen_running'")
    db.execute("UPDATE roots SET status='refine' WHERE status='refine_running'")
    db.commit()
    torch.set_num_threads(1)
    models = [load_model(p) for p in (args.parent, args.current, "checkpoints/scale100k-resnet.pt")]
    if any(
        not torch.equal(v, models[1].parent.state_dict()[k])
        for k, v in models[0].state_dict().items()
    ):
        raise ValueError("current Q model has a different frozen parent")
    futures = {}
    completed = 0
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize_worker,
        initargs=(args.parent,),
    ) as pool:
        while True:
            current = progress(db, configuration)
            atomic_json(root / "progress.json", current)
            smoke_done = args.smoke and completed >= args.smoke_tasks
            if smoke_done and not futures:
                current.update(
                    status="smoke_complete",
                    completed_tasks=completed,
                    elapsed_seconds=time.monotonic() - started,
                )
                atomic_json(root / "progress.json", current)
                return
            count = current["qualified"]
            quota_ok = all(
                current["training_strata"].get(k, 0) >= v
                for k, v in configuration["quotas"].items()
            )
            if smoke_done or (
                count.get("train", 0) >= args.target
                and count.get("valid", 0) >= args.validation
                and quota_ok
            ):
                # No new work; allow already scheduled atomic look writes to end.
                if not futures:
                    break
            else:
                pending = list(
                    db.execute(
                        "SELECT input_hash,path,status,priority,n FROM roots WHERE status IN ('screen','refine') ORDER BY priority DESC LIMIT 128"
                    )
                )
                if (
                    not pending
                    and not futures
                    and db.execute("SELECT game FROM cursor WHERE id=1").fetchone()[0]
                    >= args.max_games
                ):
                    # Keep expensive unresolved evidence. When the active queue
                    # empties, increase one selected root's budget instead of
                    # manufacturing a tie or immediately mining more deals.
                    for key, path, _, _, n in db.execute(
                        "SELECT input_hash,path,status,priority,n FROM roots WHERE status='budget_exhausted' ORDER BY priority DESC"
                    ):
                        meta = json.loads(Path(path).read_text())
                        limit = 2 ** int(
                            math.log2(
                                EVIDENCE_BYTES // (2 * len(meta["observation"]["legal_actions"]))
                            )
                        )
                        if n * 2 <= limit:
                            meta["particle_limit"] = n * 2
                            atomic_json(path, meta)
                            db.execute(
                                "UPDATE roots SET status='refine' WHERE input_hash=?", (key,)
                            )
                            db.commit()
                            pending = list(
                                db.execute(
                                    "SELECT input_hash,path,status,priority,n FROM roots WHERE input_hash=?",
                                    (key,),
                                )
                            )
                            break
                        db.execute(
                            "UPDATE roots SET status='memory_limited' WHERE input_hash=?", (key,)
                        )
                    db.commit()
                if len(pending) < args.workers * 8:
                    index = db.execute("SELECT game FROM cursor WHERE id=1").fetchone()[0]
                    if index >= args.max_games:
                        if not pending and not futures:
                            current["status"] = "mining_exhausted"
                            atomic_json(root / "progress.json", current)
                            raise RuntimeError(
                                "mining limit reached without 5k certified roots; no training"
                            )
                    else:
                        try:
                            mined = mine_game(args.start_seed + index, models)
                        except RuleClarificationRequired as exc:
                            atomic_json(
                                root / "questions" / f"game-{args.start_seed + index}.json",
                                {
                                    "seed": args.start_seed + index,
                                    "message": str(exc),
                                    "context": getattr(exc, "rollout_context", None),
                                    "used_for_training": False,
                                },
                            )
                            mined = []
                        for r in mined:
                            key = r["input_hash"]
                            if db.execute(
                                "SELECT 1 FROM roots WHERE input_hash=?", (key,)
                            ).fetchone():
                                continue
                            path = root / "candidates" / f"{key}.json"
                            atomic_json(path, r)
                            split = "valid" if r["seed"] % 10 == 9 else "train"
                            db.execute(
                                "INSERT INTO roots(input_hash,split,path,status,priority,tags,actions) VALUES(?,?,?,?,?,?,?)",
                                (
                                    key,
                                    split,
                                    str(path),
                                    "screen",
                                    r["priority"],
                                    json.dumps(r["tags"]),
                                    len(r["observation"]["legal_actions"]),
                                ),
                            )
                        db.execute("UPDATE cursor SET game=? WHERE id=1", (index + 1,))
                        db.commit()
                        continue
                if len(futures) < args.workers:

                    def priority(row, current=current):
                        meta = json.loads(Path(row[1]).read_text())
                        deficits = [
                            max(
                                0,
                                (
                                    configuration["quotas"].get(t, 0)
                                    - current["training_strata"].get(t, 0)
                                )
                                / max(1, configuration["quotas"].get(t, 0)),
                            )
                            for t in meta["tags"]
                        ]
                        return (
                            row[3]
                            + 8 * max(deficits, default=0)
                            - 1.5 * math.log2(max(1, row[4] / args.minimum))
                        )

                    for row in sorted(pending, key=priority, reverse=True)[
                        : args.workers - len(futures)
                    ]:
                        key, path, mode, _, _ = row
                        future = pool.submit(process_root, (str(root), path, mode, configuration))
                        futures[future] = (key, mode)
                        db.execute(
                            "UPDATE roots SET status=? WHERE input_hash=?", (mode + "_running", key)
                        )
                    db.commit()
            if not futures:
                continue
            done, _ = wait(futures, timeout=5, return_when=FIRST_COMPLETED)
            for future in done:
                key, mode = futures.pop(future)
                try:
                    result = future.result()
                    if mode == "screen":
                        index = db.execute(
                            "SELECT confidence_index FROM roots WHERE input_hash=?", (key,)
                        ).fetchone()[0]
                        if index is None:
                            index = (
                                db.execute("SELECT game FROM cursor WHERE id=2").fetchone()[0] + 1
                            )
                            db.execute("UPDATE cursor SET game=? WHERE id=2", (index,))
                            db.execute(
                                "UPDATE roots SET confidence_index=? WHERE input_hash=?",
                                (index, key),
                            )
                            db.commit()
                        candidate = Path(
                            db.execute(
                                "SELECT path FROM roots WHERE input_hash=?", (key,)
                            ).fetchone()[0]
                        )
                        meta = json.loads(candidate.read_text())
                        meta.update(
                            confidence_index=index,
                            root_alpha=args.alpha / (index * (index + 1)),
                            particle_limit=meta.get(
                                "particle_limit",
                                min(
                                    args.maximum,
                                    2
                                    ** int(
                                        math.log2(
                                            EVIDENCE_BYTES
                                            // (2 * len(meta["observation"]["legal_actions"]))
                                        )
                                    ),
                                ),
                            ),
                        )
                        atomic_json(candidate, meta)
                    db.execute(
                        "UPDATE roots SET status=?,priority=?,n=? WHERE input_hash=?",
                        (result["status"], result["priority"], result["n"], key),
                    )
                except SamplingFailure as exc:
                    db.execute(
                        "UPDATE roots SET status='sampling_failed' WHERE input_hash=?", (key,)
                    )
                    print(
                        json.dumps({"sampling_failed": key, "mode": mode, "message": str(exc)}),
                        flush=True,
                    )
                except RuleClarificationRequired as exc:
                    db.execute(
                        "UPDATE roots SET status='rule_unresolved' WHERE input_hash=?", (key,)
                    )
                    atomic_json(
                        root / "questions" / f"{key}.json",
                        {
                            "input_hash": key,
                            "mode": mode,
                            "message": str(exc),
                            "context": getattr(exc, "rollout_context", None),
                            "used_for_training": False,
                        },
                    )
                    print(
                        json.dumps({"rule_unresolved": key, "mode": mode, "message": str(exc)}),
                        flush=True,
                    )
                db.commit()
                completed += 1
                summary = progress(db, configuration)
                summary.update(
                    completed_tasks=completed, elapsed_seconds=time.monotonic() - started
                )
                atomic_json(root / "progress.json", summary)
                print(json.dumps(summary, ensure_ascii=False), flush=True)
    manifest = finish_manifest(db, root, configuration)
    audit_dataset(root)
    status = progress(db, configuration)
    status["status"] = "ready"
    atomic_json(root / "progress.json", status)
    if args.train_after:
        training_gate(manifest, args.target, args.validation)
        python = str(Path(".venv-cuda/bin/python").resolve())
        output = "checkpoints/activeq-resnet.pt"
        status.update(status="training", training_started=True)
        atomic_json(root / "progress.json", status)
        subprocess.run(
            [
                python,
                "-m",
                "zimortal.training.cashq",
                "train",
                "--data",
                str(root),
                "--parent",
                args.parent,
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
                args.parent,
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
        status.update(
            status="complete", training_started=True, report=str(root / "comparison.json")
        )
        atomic_json(root / "progress.json", status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "status", "audit"])
    parser.add_argument("--data", default="data/generated/active-v2")
    parser.add_argument("--parent", default="checkpoints/huxi-resnet.pt")
    parser.add_argument("--current", default="checkpoints/cashq-regret.gated.pt")
    parser.add_argument("--target", type=int, default=5000)
    parser.add_argument("--validation", type=int, default=500)
    parser.add_argument("--minimum", type=int, default=128)
    parser.add_argument("--maximum", type=int, default=1048576)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--tie-cash", type=float, default=1)
    parser.add_argument("--scope", choices=["top_set", "full_order"], default="top_set")
    parser.add_argument("--max-attempts", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--start-seed", type=int, default=2000000)
    parser.add_argument("--max-games", type=int, default=100000)
    parser.add_argument("--train-after", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-tasks", type=int, default=12)
    args = parser.parse_args()
    if args.command == "status":
        print((Path(args.data) / "progress.json").read_text())
    elif args.command == "audit":
        print(json.dumps(audit_dataset(args.data)))
    else:
        if (
            min(
                args.target,
                args.validation,
                args.workers,
                args.max_attempts,
                args.smoke_tasks,
                args.max_games,
            )
            < 1
        ):
            parser.error("positive counts required")
        if args.target < 5000 and not args.smoke:
            parser.error("at least 5000 certified training roots required")
        if args.smoke and args.train_after:
            parser.error("smoke data cannot train a model")
        if args.start_seed < 0 or (
            args.train_after
            and args.start_seed < 3002100
            and args.start_seed + args.max_games > 3000000
        ):
            parser.error("mining range must exclude independent evaluation seeds")
        try:
            run(args)
        except Exception as exc:
            path = Path(args.data) / "progress.json"
            status = json.loads(path.read_text()) if path.exists() else {}
            status.update(status="failed", error=str(exc), timestamp=time.time())
            atomic_json(path, status)
            raise


if __name__ == "__main__":
    main()
