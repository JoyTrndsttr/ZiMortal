import hashlib
import json
import sqlite3
from dataclasses import asdict, replace

import numpy as np
import pytest

from zimortal.engine import RuleEngine
from zimortal.training import adaptive, envelope_evidence
from zimortal.training.active import atomic_json, atomic_npz, serialize_observation


def test_legacy_certificates_stay_byte_identical_and_dispatch_original_bound(tmp_path, monkeypatch):
    engine = RuleEngine()
    obs = engine.observation(engine.new_game(118), 0)
    (tmp_path / "labels").mkdir()
    (tmp_path / "candidates").mkdir()
    atomic_json(tmp_path / "config.json", {"adaptive": asdict(adaptive.AdaptiveConfig())})
    atomic_json(tmp_path / "candidates/root.json", {"observation": serialize_observation(obs)})
    atomic_json(tmp_path / "labels/root.json", {"qualified": True})
    atomic_npz(tmp_path / "labels/root.npz", sentinel=np.array([123]))
    # Explicit whitelist models an already re-audited legacy family. Default
    # migration creates an empty whitelist because historical math was unsafe.
    atomic_json(tmp_path / "legacy-certified-roots.json", ["root"])
    before_json = (tmp_path / "labels/root.json").read_bytes()
    before_npz = (tmp_path / "labels/root.npz").read_bytes()
    envelope_evidence.migrate(tmp_path, write=True)
    assert (tmp_path / "labels/root.json").read_bytes() == before_json
    assert (tmp_path / "labels/root.npz").read_bytes() == before_npz
    envelope_evidence.load_legacy(tmp_path)
    monkeypatch.setattr(envelope_evidence, "original_bounds", lambda _: (-100, 100))
    monkeypatch.setattr(envelope_evidence, "payoff_bounds", lambda _: (-10, 10))
    assert envelope_evidence.dataset_bounds(obs) == (-100, 100)
    assert envelope_evidence.dataset_bounds(replace(obs, remaining_tiles=0)) == (-10, 10)


@pytest.mark.parametrize("revoke", [False, True])
@pytest.mark.parametrize("excluded", [False, True])
def test_recertification_keeps_raw_particles_alpha_and_excluded_roots(
    tmp_path, monkeypatch, excluded, revoke
):
    engine = RuleEngine()
    state = engine.new_game(118)
    while len(engine.legal_actions(state)) == 1:
        state = engine.step(state, engine.legal_actions(state)[0])
    obs = engine.observation(state, state.turn)
    obs = replace(obs, legal_actions=obs.legal_actions[:2])
    assert len(obs.legal_actions) == 2
    (tmp_path / "labels").mkdir()
    (tmp_path / "candidates").mkdir()
    raw = np.zeros((1024, 2), np.int16)
    raw_hash = hashlib.sha256(raw.tobytes()).hexdigest()
    atomic_npz(tmp_path / "labels/root.npz", outcomes=raw, outcomes_hash=raw_hash)
    atomic_json(tmp_path / "config.json", {"adaptive": asdict(adaptive.AdaptiveConfig())})
    atomic_json(tmp_path / "candidates/root.json", {"observation": serialize_observation(obs)})
    atomic_json(
        tmp_path / "labels/root.json",
        {"alpha": 0.001, "qualified": revoke, "tags": ["late"], "rollouts_per_action": 1024},
    )
    with sqlite3.connect(tmp_path / "queue.sqlite") as db:
        db.execute("CREATE TABLE roots (input_hash TEXT, status TEXT)")
        db.execute(
            "INSERT INTO roots VALUES ('root', ?)",
            ("rule_unresolved" if excluded else "qualified" if revoke else "deferred",),
        )
    monkeypatch.setattr(adaptive, "payoff_bounds", lambda _: (-100, 100))
    monkeypatch.setattr(
        envelope_evidence, "payoff_bounds", lambda _: (-1000, 1000) if revoke else (-1, 1)
    )
    envelope_evidence.migrate(tmp_path, write=True)
    meta = json.loads((tmp_path / "labels/root.json").read_text())
    assert meta["alpha"] == 0.001
    assert meta["qualified"] == (not revoke)
    with np.load(tmp_path / "labels/root.npz") as saved:
        np.testing.assert_array_equal(saved["outcomes"], raw)
        assert str(saved["outcomes_hash"]) == raw_hash
    with sqlite3.connect(tmp_path / "queue.sqlite") as db:
        assert db.execute("SELECT status FROM roots").fetchone()[0] == (
            "rule_unresolved" if excluded else "deferred" if revoke else "qualified"
        )
