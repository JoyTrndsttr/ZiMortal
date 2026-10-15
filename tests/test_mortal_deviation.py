import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from zimortal.engine import Action, ActionType
from zimortal.model.encoding import encode_action
from zimortal.training.mortal_deviation import DeviationPolicy
from zimortal.training.mortal_deviation_stats import P_EDGES, bin_index, cluster_cohort

BEHAVIOR = {"discard": 1.5, "chi/pass": 1.5, "peng/pass": 2.5,
            "hu/pass": 1., "hu_pass_probability": .02}


class Parent(nn.Module):
    feature_version = "huxi"

    def forward(self, x, a, mask):
        logits = torch.zeros_like(mask, dtype=torch.float).masked_fill(~mask, -torch.inf)
        return logits, torch.zeros(len(mask)), {}


class FixedQ(nn.Module):
    def forward_q(self, x, a, mask):
        scores = torch.arange(mask.shape[1], dtype=torch.float).expand_as(mask).masked_fill(~mask, -torch.inf)
        return scores, torch.zeros(len(mask))


def toy_batch(policy, seeds):
    actions = [Action(ActionType.DISCARD, 0, tile=i) for i in range(2)]
    a = torch.from_numpy(np.stack([encode_action(v, 0) for v in actions]))[None].repeat(len(seeds), 1, 1)
    specs = [(s, 0, 0, 0) for s in seeds]
    policy.prepare(specs)
    score = policy(torch.zeros(len(seeds), 55, 20), a, torch.ones(len(seeds), 2, dtype=torch.bool))[0]
    return specs, actions, score.argmax(-1).tolist()


def test_budget_tracks_games_not_reused_batch_slots_and_counts_execution_only():
    p = DeviationPolicy(FixedQ(), Parent(), BEHAVIOR, 1).eval()
    specs, actions, choices = toy_batch(p, [11, 12])
    assert choices == [1, 1] and not p.counts
    for i, s in enumerate(specs):
        r = p.record(SimpleNamespace(legal_actions=actions), i, choices[i], s, 1)
        assert r["deviation_count_after"] == 1
    specs, actions, choices = toy_batch(p, [12, 13])
    assert choices == [0, 1]
    for i, s in enumerate(specs):
        r = p.record(SimpleNamespace(legal_actions=actions), i, choices[i], s, 2)
        assert r["blocked_by_budget"] == (s[0] == 12)
    assert p.counts == {(11, 0): 1, (12, 0): 1, (13, 0): 1}


def test_context_required_for_budgeted_inference():
    p = DeviationPolicy(FixedQ(), Parent(), BEHAVIOR, 1)
    with pytest.raises(RuntimeError):
        p(torch.zeros(1, 55, 20), torch.zeros(1, 2, 10), torch.ones(1, 2, dtype=torch.bool))
    with pytest.raises(ValueError):
        p.prepare([(11, 0, 0, 0), (11, 0, 0, 0)])


def test_conditional_bootstrap_weights_selected_games_not_equal_seed_means():
    games = [{"seed": 1, "seat": 0, "cash": 10}, {"seed": 1, "seat": 1, "cash": 20},
             {"seed": 2, "seat": 0, "cash": -30}]
    baseline = {(g["seed"], g["seat"]): {"cash": 0} for g in games}
    s = cluster_cohort(games, baseline, bootstrap=2000)
    assert s["paired_delta"] == 0
    assert s["seed_bootstrap_95"] == [-30, 15]
    assert cluster_cohort([], baseline)["paired_delta"] is None
    assert bin_index(.1, P_EDGES) == 1
    assert bin_index(1., P_EDGES) == len(P_EDGES) - 2
    with pytest.raises(ValueError):
        bin_index(float("nan"), P_EDGES)


@pytest.mark.parametrize("budget", [0, 1, 2, None])
def test_complete_frozen_budget_games_audit_against_actual_history(budget):
    from zimortal.engine import GameState
    from zimortal.training.mortal_deviation_experiment import audit_batch
    from zimortal.training.mortal_deviation_stats import deviation_statistics
    from zimortal.training.online_rl import play

    p = DeviationPolicy(FixedQ(), Parent(), BEHAVIOR, budget).eval()
    pending, records, counts = {}, [], {}
    def sink(e):
        pending.update(state=GameState.deserialize(e["state"]), key=(e["seed"], e["seat"]))
    def record(batch):
        key = pending["key"]
        counts[key] = audit_batch(batch, pending, budget)
        records.extend(batch)
    rows, games, failures = play(p, [p.parent], [(1159999900+i, i % 3, i % 3, 0) for i in range(3)],
                               replay_sink=sink, decision_context=p.prepare,
                               decision_recorder=p.record, decision_sink=record)
    assert not rows and not failures and len(games) == 3
    assert all(n <= budget for n in counts.values()) if budget is not None else True
    assert sum(counts.values()) == sum(r["changed"] for r in records)
    if budget:
        assert sum(counts.values()) > 0
    baseline = {(g["seed"], g["seat"]): {"cash": g["cash"]} for g in games}
    json.dumps(deviation_statistics(records, games, baseline, counts))
