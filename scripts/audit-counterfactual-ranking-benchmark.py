"""Replay every locked world/action with ordinary RuleEngine and frozen policies."""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

from zimortal.engine import GameState, RuleClarificationRequired, RuleEngine
from zimortal.training.active import atomic_json, serialize_observation
from zimortal.training.counterfactual_rollout import (
    QUOTAS,
    WORLDS,
    assert_frozen,
    continue_root,
    ev_statistics,
    make_world,
    pick_roots,
    root_predictions,
    seeded,
    state_digest,
)
from zimortal.training.mortal_credit_diagnostics import blob
from zimortal.training.mortal_sarsa import parameter_hash
from zimortal.training.online_rl import digest


def runner():
    spec = importlib.util.spec_from_file_location(
        "ranking_runner", "scripts/run-counterfactual-ranking-benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(directory, device):
    m = runner()
    root = Path(directory)
    _config, entries = m.verify_lock(root)
    result_sha = digest(root / "result.json")
    outputs = json.loads((root / "output-sha256.json").read_text())
    for p, sha in outputs.items():
        assert digest(p) == sha, p
    selected = pick_roots(json.loads(m.POOL.read_text()))
    assert [r["locator"] for r in selected] == json.loads(
        (root / "selected-locators.json").read_text()
    )
    assert {f: sum(e["family"] == f for e in entries) for f in QUOTAS} == QUOTAS
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    models, baseline, behavior, anchor = m.load_bundle(device)
    before = {k: parameter_hash(v) for k, v in (models | {"baseline": baseline}).items()}
    done = branches = steps = 0
    started = time.perf_counter()
    for e in entries:
        payload = m.read_gz(root / e["file"])
        state = GameState.deserialize(payload["root"])
        player = payload["locator"][5]
        engine = RuleEngine()
        obs = engine.observation(state, player)
        assert serialize_observation(obs) == payload["observation"]
        assert state_digest(state) == payload["root_sha256"]
        assert json.loads(blob(obs.legal_actions)) == payload["legal_actions"]
        initial = GameState.deserialize(payload["initial"])
        assert engine.replay(initial, state.history).serialize() == state.serialize()
        for j in range(WORLDS):
            world, _record = make_world(state, e["id"], j)
            alternative_initial = initial.clone()
            alternative_initial.deck[: len(world.deck)] = world.deck
            rebuilt = engine.replay(alternative_initial, state.history)
            assert rebuilt.serialize() == world.serialize()  # all shuffled walls are reachable
        pred, p, parent = root_predictions(state, models, baseline, behavior, device=device)
        np.testing.assert_allclose(p, payload["p_reference"], atol=1e-7, rtol=1e-6)
        assert parent == payload["baseline_action"]
        for cp in models:
            for mode in ("q", "anchor"):
                np.testing.assert_allclose(
                    pred[cp][mode], payload["predictions"][cp][mode], atol=1e-5, rtol=1e-5
                )
        saved = m.read_gz(root / f"{e['id']}-rollouts.json.gz")
        try:
            values, evidence = continue_root(
                state, e["id"], anchor, baseline, engine=RuleEngine(), device=device
            )
            assert "failure" not in saved
            np.testing.assert_array_equal(values, saved["outcomes"])
            assert evidence == saved["evidence"]  # seeds, wall, every action, terminal SHA and cash
            assert ev_statistics(values, seeded(e["id"], "statistics")) == saved["result"]["ev"]
            branches += values.size
            steps += evidence["counts"]["future_steps"]
        except RuleClarificationRequired as exc:
            assert saved.get("failure") == str(exc)
        done += 1
        atomic_json(
            root / "audit-progress.json",
            {
                "roots_verified": done,
                "branches_reexecuted": branches,
                "future_steps": steps,
                "seconds": time.perf_counter() - started,
            },
        )
    assert_frozen(models | {"baseline": baseline}, before)
    m.verify_lock(root)
    for p, sha in outputs.items():
        assert digest(p) == sha
    # Exact aggregation rerun from saved outcomes, no labels added to training.
    report = m.analyze(root, save=False)
    assert report == json.loads((root / "result.json").read_text())
    assert digest(root / "result.json") == result_sha
    atomic_json(
        root / "audit-result.json",
        {
            "status": "complete_audit_stop",
            "roots_verified": done,
            "branches_reexecuted": branches,
            "future_steps": steps,
            "seconds": time.perf_counter() - started,
            "all_worlds_reachable": True,
            "all_seeds_CRN_verified": True,
            "every_policy_action_suffix_terminal_cash_verified": True,
            "ranking_aggregation_verified": True,
            "checkpoint_BN_sha_unchanged": True,
            "source_sha_verified": True,
            "unique_new_particles_in_audit": 0,
            "training_updates": 0,
            "baseline_replaced": False,
            "result_sha256": digest(root / "result.json"),
            "source_sha256": digest("scripts/audit-counterfactual-ranking-benchmark.py"),
            "valid_states": report["states"],
        },
    )
    print("All fixed worlds/actions reproduced; no training updates", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="data/generated/counterfactual-ranking-v1")
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = parser.parse_args()
    run(args.directory, args.device)
