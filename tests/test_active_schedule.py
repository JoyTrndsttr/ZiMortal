"""Budget lane fairness and evidence-preserving scheduling migrations."""

import json
import sqlite3

import pytest

pytest.importorskip("torch")
from zimortal.training import active_schedule as schedule


def queue():
    db = sqlite3.connect(":memory:")
    db.execute(
        "CREATE TABLE roots(input_hash TEXT,path TEXT,status TEXT,priority REAL,n INTEGER,tags TEXT,actions INTEGER,split TEXT)"
    )
    return db


def insert(db, key, status, n, score=10, split="train"):
    db.execute(
        "INSERT INTO roots VALUES(?,?,?,?,?,?,?,?)",
        (key, "unused", status, score, n, json.dumps(["chi_bi"]), 2, split),
    )


def config():
    return {"quotas": {"chi_bi": 1000}, "validation": 500, "adaptive": {"minimum": 128}}


def test_deep_work_cannot_block_fresh_decisions_even_with_greater_priority():
    db = queue()
    insert(db, "expensive", "deferred", 16384, score=1000)
    insert(db, "new", "screen", 0)
    insert(db, "fresh", "refine", 128)
    current = {"qualified": {}, "training_strata": {}}
    for ticket in range(8):
        row = schedule.choose_job(db, current, config(), ticket, True, 65536)
        assert row[0] in ("new", "fresh")
    assert schedule.choose_job(db, current, config(), 2, False, 65536)[0] == "expensive"


def test_deep_cap_and_validation_deficit_affect_selection():
    db = queue()
    insert(db, "capped", "deferred", 65536, score=1000)
    insert(db, "train", "deferred", 1024)
    insert(db, "valid", "deferred", 1024, split="valid")
    current = {"qualified": {}, "training_strata": {}}
    assert schedule.choose_job(db, current, config(), 2, False, 65536)[0] == "valid"
    current["qualified"]["valid"] = 500
    assert schedule.choose_job(db, current, config(), 2, False, 65536)[0] == "train"


def test_copy_preserves_raw_evidence_config_and_confidence_indices(tmp_path, monkeypatch):
    source = tmp_path / "old"
    source.mkdir()
    (source / "candidates").mkdir()
    (source / "labels").mkdir()
    (source / "config.json").write_text(json.dumps({"statistics": "immutable"}))
    (source / "progress.json").write_text(json.dumps({"qualified": {"train": 2}}))
    (source / "scheduler.json").write_text('{"source_sha256":"old scheduler"}')
    (source / "labels" / "root.npz").write_bytes(b"lossless original samples")
    (source / "candidates" / "root.json").write_text("{}")
    with sqlite3.connect(source / "queue.sqlite") as db:
        db.execute(
            "CREATE TABLE roots(input_hash TEXT,path TEXT,n INTEGER,confidence_index INTEGER,status TEXT)"
        )
        db.execute(
            "INSERT INTO roots VALUES(?,?,?,?,?)",
            ("root", str(source / "candidates" / "root.json"), 65536, 7, "qualified"),
        )
    monkeypatch.setattr(schedule, "verify_producer", lambda _: None)
    target = tmp_path / "new"
    schedule.copy_evidence(source, target)
    assert (target / "config.json").read_bytes() == (source / "config.json").read_bytes()
    assert (target / "labels" / "root.npz").read_bytes() == b"lossless original samples"
    with sqlite3.connect(target / "queue.sqlite") as db:
        assert db.execute("SELECT n,confidence_index,status,path FROM roots").fetchone() == (
            65536,
            7,
            "qualified",
            str(target / "candidates" / "root.json"),
        )
    (target / "labels" / "root.npz").write_bytes(b"new version")
    assert (source / "labels" / "root.npz").read_bytes() == b"lossless original samples"
    lineage = json.loads((target / "lineage.json").read_text())
    assert lineage["confidence_indices_preserved"]
    assert lineage["source_scheduler"]["source_sha256"] == "old scheduler"
    assert not (target / "scheduler.json").exists()


def test_live_coordinator_cannot_be_copied(tmp_path, monkeypatch):
    source = tmp_path / "live"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "progress.json").write_text('{"process_id":123}')
    monkeypatch.setattr(schedule, "verify_producer", lambda _: None)
    monkeypatch.setattr(schedule.os, "kill", lambda *args: None)
    with pytest.raises(ValueError, match="stop"):
        schedule.copy_evidence(source, tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_evidence_memory_cap_cannot_cause_endless_same_look_promotion():
    db = queue()
    insert(db, "memory_limited", "deferred", 32768, score=1000)
    db.execute("UPDATE roots SET actions=2000 WHERE input_hash='memory_limited'")
    current = {"qualified": {}, "training_strata": {}}
    assert schedule.choose_job(db, current, config(), 2, False, 1048576) is None
