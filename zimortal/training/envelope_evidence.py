"""Recompute derived confidence evidence; never change independent cash particles."""

import argparse
import hashlib
import json
import math
import sqlite3
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import active, adaptive
from .adaptive import payoff_bounds as original_bounds
from .payoff_envelope import payoff_bounds
from .rollout_engine import initialize_worker as initialize_cached_worker

LEGACY_OBSERVATIONS = set()


def prepare_legacy(root):
    root = Path(root)
    path = root / "legacy-certified-roots.json"
    if not path.exists():
        keys = [
            p.stem
            for p in (root / "labels").glob("*.json")
            if json.loads(p.read_text())["qualified"]
        ]
        active.atomic_json(path, sorted(keys))
    return set(json.loads(path.read_text()))


def load_legacy(root):
    global LEGACY_OBSERVATIONS
    root = Path(root)
    LEGACY_OBSERVATIONS = {
        active.deserialize_observation(
            json.loads((root / "candidates" / f"{key}.json").read_text())["observation"]
        )
        for key in prepare_legacy(root)
    }


def dataset_bounds(obs):
    return original_bounds(obs) if obs in LEGACY_OBSERVATIONS else payoff_bounds(obs)


def initialize_worker(parent, root):
    initialize_cached_worker(parent)
    load_legacy(root)
    adaptive.payoff_bounds = dataset_bounds


def analyze(raw, obs, config):
    trace, previous, result, qualified = [], None, None, False
    earliest = None
    for look in range(int(math.log2(len(raw) // config.minimum)) + 1):
        n = config.minimum << look
        result = adaptive.relations(raw[:n], obs, config, look)
        signature = (
            result["relations"].tobytes()
            if config.scope == "full_order"
            else tuple(result["optimal"])
        )
        qualified = result["resolved"] and previous == signature
        if qualified and earliest is None:
            earliest = n
        trace.append({k: v for k, v in result.items() if k not in ("intervals", "relations")})
        previous = signature if result["resolved"] else None
    return result, trace, bool(qualified), earliest


def migrate(root, *, write=False, limit=None):
    root = Path(root)
    configuration = json.loads((root / "config.json").read_text())
    envelope_hash = active.digest("zimortal/training/payoff_envelope.py")
    legacy = prepare_legacy(root) if write else set()
    rows = []
    paths = list((root / "labels").glob("*.json"))
    # Diagnostic selection: largest existing looks; no new alpha allocation.
    if limit is not None:
        paths.sort(key=lambda p: json.loads(p.read_text())["rollouts_per_action"], reverse=True)
        paths = paths[:limit]
    db = sqlite3.connect(root / "queue.sqlite")
    for path in paths:
        meta = json.loads(path.read_text())
        if write and path.stem in legacy:
            continue
        if write and meta.get("payoff_envelope_sha256") == envelope_hash:
            continue
        obs = active.deserialize_observation(
            json.loads((root / "candidates" / path.name).read_text())["observation"]
        )
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as saved:
            # One root at a time, bounded by the existing 128 MiB evidence cap.
            arrays = {k: saved[k] for k in saved.files}
        raw = arrays["outcomes"]
        if str(arrays["outcomes_hash"]) != hashlib.sha256(raw.tobytes()).hexdigest():
            raise ValueError("raw evidence checksum mismatch")
        config = adaptive.AdaptiveConfig(**{**configuration["adaptive"], "alpha": meta["alpha"]})
        with patch.object(adaptive, "payoff_bounds", payoff_bounds):
            result, trace, qualified, earliest = analyze(raw, obs, config)
        old_bounds = adaptive.payoff_bounds(obs)
        rows.append(
            {
                "root": path.stem,
                "tags": meta["tags"],
                "n": len(raw),
                "old_qualified": meta["qualified"],
                "new_qualified": qualified,
                "earliest_new_qualified_n": earliest,
                "old_bounds": old_bounds,
                "new_bounds": result["payoff_bounds_cash"],
            }
        )
        if write:
            optimal = np.array([i in result["optimal"] for i in range(raw.shape[1])])
            soft = np.zeros(raw.shape[1], np.float32)
            soft[optimal] = 1 / optimal.sum()
            arrays.update(
                intervals=result["intervals"],
                relations=result["relations"],
                optimal=optimal,
                soft=soft,
            )
            active.atomic_npz(path.with_suffix(".npz"), **arrays)
            meta.update(
                qualified=qualified,
                trace=trace,
                payoff_bounds_cash=result["payoff_bounds_cash"],
                sha256=active.digest(path.with_suffix(".npz")),
                payoff_envelope_sha256=envelope_hash,
                pair_relations_sha256=hashlib.sha256(result["relations"].tobytes()).hexdigest(),
            )
            active.atomic_json(path, meta)
            # Roots excluded after undefined rules or sampling failure remain excluded.
            if qualified:
                db.execute(
                    "UPDATE roots SET status='qualified' WHERE input_hash=? AND status IN ('qualified','deferred','refine','refine_running','deep_running')",
                    (path.stem,),
                )
                db.commit()
        if len(rows) % 250 == 0:
            print(json.dumps({"envelope_recomputed": len(rows)}, ensure_ascii=False), flush=True)
    db.close()
    report = {
        "inherited_qualified_unchanged": len(legacy),
        "payoff_envelope_sha256": envelope_hash,
        "write": write,
        "roots": rows,
        "raw_particles_unchanged": True,
        "old_qualified": sum(r["old_qualified"] for r in rows),
        "new_qualified": sum(r["new_qualified"] for r in rows),
    }
    active.atomic_json(
        root / ("envelope-migration.json" if write else "envelope-diagnostic.json"), report
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/generated/active-v5")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        audit_partial(args.data, args.limit)
        return
    report = migrate(args.data, limit=args.limit)
    print(json.dumps({k: v for k, v in report.items() if k != "roots"}), flush=True)


def audit_partial(root, limit=64):
    """Reproduce both interval families before the full 5k training audit."""
    root = Path(root)
    load_legacy(root)
    configuration = json.loads((root / "config.json").read_text())
    with sqlite3.connect(root / "queue.sqlite") as db:
        keys = [
            k
            for (k,) in db.execute(
                "SELECT input_hash FROM roots WHERE status='qualified' ORDER BY input_hash"
            )
        ]
    legacy = prepare_legacy(root)
    selected = [k for k in keys if k in legacy][: limit // 2] + [
        k for k in keys if k not in legacy
    ][: limit // 2]
    for key in selected:
        meta = json.loads((root / "labels" / f"{key}.json").read_text())
        path = root / meta["file"]
        if active.digest(path) != meta["sha256"]:
            raise ValueError("audit evidence changed")
        obs = active.deserialize_observation(
            json.loads((root / "candidates" / f"{key}.json").read_text())["observation"]
        )
        with np.load(path, allow_pickle=False) as saved:
            raw = saved["outcomes"]
            if hashlib.sha256(raw.tobytes()).hexdigest() != str(saved["outcomes_hash"]):
                raise ValueError("audit raw particles changed")
            config = adaptive.AdaptiveConfig(
                **{**configuration["adaptive"], "alpha": meta["alpha"]}
            )
            with patch.object(adaptive, "payoff_bounds", dataset_bounds):
                result, _, qualified, _ = analyze(raw, obs, config)
            if not qualified:
                raise ValueError("audit two-look certification failed")
            np.testing.assert_allclose(result["intervals"], saved["intervals"], rtol=0, atol=1e-10)
            np.testing.assert_array_equal(result["relations"], saved["relations"])
            np.testing.assert_array_equal(np.flatnonzero(saved["optimal"]), result["optimal"])
    report = {
        "checked": len(selected),
        "legacy_checked": sum(k in legacy for k in selected),
        "new_envelope_checked": sum(k not in legacy for k in selected),
        "training_gate_passed": False,
    }
    active.atomic_json(root / "envelope-audit.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
