import copy

import numpy as np
import pytest
import torch

from zimortal.engine import RuleEngine
from zimortal.model.network import PolicyValueNet
from zimortal.training.counterfactual_rollout import (
    QUOTAS,
    WORLDS,
    continue_root,
    ev_statistics,
    make_world,
    pick_roots,
    ranking,
    seeded,
    state_digest,
    world_record,
)
from zimortal.training.mortal_sarsa import parameter_hash
from zimortal.training.runtime import predict


def test_sampling_uses_only_family_locators_and_fixed_quotas():
    rows = [
        {"family": family, "locator": [0, i], "cash": i} for family in QUOTAS for i in range(250)
    ]
    alternate = copy.deepcopy(rows)
    for r in alternate:
        r["cash"] = -1000000
    a, b = pick_roots(rows), pick_roots(alternate)
    assert [r["locator"] for r in a] == [r["locator"] for r in b]
    assert {f: sum(r["family"] == f for r in a) for f in QUOTAS} == QUOTAS


def test_hidden_hands_fixed_crn_wall_reachable_and_observation_unchanged():
    engine = RuleEngine()
    initial = engine.new_game(118, 0)
    state = initial
    while len(engine.legal_actions(state)) == 1:
        state = engine.step(state, engine.legal_actions(state)[0])
    player = engine.legal_actions(state)[0].player
    original = state_digest(state)
    for j in (0, 1, 2):
        world, record = make_world(state, "toy", j)
        duplicate, same = make_world(state, "toy", j)
        assert record == same and state_digest(world) == state_digest(duplicate)
        assert [p.hand for p in world.players] == [p.hand for p in state.players]
        assert engine.observation(world, player) == engine.observation(state, player)
        origin = initial.clone()
        origin.deck[: len(world.deck)] = world.deck
        assert engine.replay(origin, state.history).serialize() == world.serialize()
        assert record["hidden_state_seed_usage"].startswith("unused")
        assert world_record("toy", j)["wall_seed"] == record["wall_seed"]
    assert state_digest(state) == original


def test_ev_paired_ci_ties_and_ranking_do_not_certify_zero_as_best():
    values = np.column_stack(
        [np.arange(WORLDS) * 5, np.arange(WORLDS) * 5 - 10, np.arange(WORLDS) * 5]
    )
    ev = ev_statistics(values, seeded("toy", "stats"))
    assert ev["best_set"] == [0, 2] and not ev["clear"]
    pair = next(p for p in ev["pairs"] if p["i"] == 0 and p["j"] == 1)
    assert pair["mean"] == 10 and pair["se"] == 0 and pair["ci95"] == [10, 10]
    r = ranking([0, 1, 2], ev, values)
    assert r["best_action_hit_rate"] == 1 and r["top1_accuracy"] == 0
    assert r["regret_yuan"] == 0 and r["pairwise_accuracy"] == 0.5
    correct = ev_statistics(np.column_stack([values[:, 0] + 20, values[:, 0]]), 7)
    assert correct["clear"] and correct["top_second_ci95"] == [20, 20]


def test_empty_tied_pairs_have_no_fabricated_accuracy():
    values = np.zeros((WORLDS, 3))
    ev = ev_statistics(values, 6)
    r = ranking([3, 2, 1], ev, values)
    assert r["pairwise_accuracy"] is None and r["spearman"] is None
    assert r["kendall_tau_b"] is None and not ev["clear"]
    assert r["best_action_hit_rate"] == 1


def test_frozen_true_rollouts_are_repeatable_without_parameter_updates():
    # A late real state gives a small, meaningful full-terminal engine regression.
    engine = RuleEngine()
    state = engine.new_game(46, 0)
    torch.manual_seed(5)
    model = PolicyValueNet("mlp", 8, feature_version="huxi")
    model.eval().requires_grad_(False)
    for _ in range(400):
        actions = engine.legal_actions(state)
        if len(state.deck) <= 2 and len(actions) >= 2:
            break
        if state.terminal:
            pytest.skip("test seed terminated before late multi-action state")
        action = (
            actions[0]
            if len(actions) == 1
            else actions[
                int(predict(model, engine.observation(state, actions[0].player))[0].argmax())
            ]
        )
        state = engine.step(state, action)
    else:
        raise AssertionError("no late decision")
    before = parameter_hash(model)
    a, evidence = continue_root(state, "test", model, model, device="cpu")
    b, again = continue_root(state, "test", model, model, device="cpu")
    np.testing.assert_array_equal(a, b)
    assert evidence == again and a.shape[0] == 32
    assert parameter_hash(model) == before and all(p.grad is None for p in model.parameters())


def test_read_only_report_cluster_interval_uses_paired_games():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "benchmark_report", Path("scripts/report-counterfactual-ranking-benchmark.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [
        {"locator": ["LR-B", 1, 0, 0, game], "ranking": {"a": {"hit": 1.0}, "b": {"hit": 0.0}}}
        for game in (1, 1, 2, 3)
    ]
    result = module.cluster_difference(rows, "a", "b", "hit", count=128)
    assert result["mean"] == 1 and result["ci95"] == [1, 1]
    assert module.cluster_difference([], "a", "b", "hit")["mean"] is None
