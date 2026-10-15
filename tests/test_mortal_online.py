import json

import numpy as np
import pytest
import torch
from test_mortal_deviation import BEHAVIOR, Parent

from zimortal.model.mortal_q import MortalQNet
from zimortal.training.mortal_data import audit_dataset, load_rows, save_shard
from zimortal.training.mortal_extraction import ExtractionPolicy
from zimortal.training.mortal_online import collect, learning_rate, make_optimizer, mc_loss, update
from zimortal.training.mortal_online_experiment import seed_specs, stop_reason
from zimortal.training.mortal_sarsa import parameter_hash


def test_mc_only_ignores_unselected_q_except_structural_dueling_dependence():
    q = torch.tensor([[1., 100., -torch.inf]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    loss = mc_loss(q, torch.tensor([0]), torch.tensor([3.]), mask)
    assert loss.item() == 2
    loss.backward()
    assert q.grad.tolist() == [[-2., 0., 0.]]  # no CQL/entropy/pairwise term
    with pytest.raises(ValueError):
        mc_loss(q, torch.tensor([2]), torch.tensor([3.]), mask)


def test_finite_seed_budget_no_train_eval_overlap_and_stopping():
    seen = set()
    for g in range(1, 4):
        train, test = seed_specs(g), seed_specs(g, True)
        assert len(train) == 3000 and len(test) == 1500
        a, b = {s[0] for s in train}, {s[0] for s in test}
        assert a.isdisjoint(b) and seen.isdisjoint(a | b)
        seen |= a | b
        assert {(d, seat) for _, d, seat, _ in train} == {(d, s) for d in range(3) for s in range(3)}
    assert stop_reason({'seed_bootstrap_95': [-2, -1]}, 1, 1) == 'clear_degradation_rollback'
    assert stop_reason({'seed_bootstrap_95': [-1, 1]}, 1, 1) is None
    assert 'rollback' in stop_reason({'seed_bootstrap_95': [-1, 1]}, 3, 3)
    assert learning_rate(0) == pytest.approx(3e-6)
    assert learning_rate(20) == pytest.approx(3e-5)
    assert learning_rate(320) == pytest.approx(3e-6)


def test_real_visited_mc_rows_shard_audit_and_persistent_optimizer(tmp_path):
    torch.manual_seed(8)
    model, parent = MortalQNet(width=4, latent=16, blocks=1), Parent().requires_grad_(False)
    policy = ExtractionPolicy(model, parent, BEHAVIOR, {"mode": "anchor", "lambda": .3})
    parent_hash, actor_hash = parameter_hash(parent), parameter_hash(model)
    games, records, data = collect(policy, parent, [(1199999990 + i, i % 3, i % 3, 0) for i in range(3)], tmp_path, 'toy', training_rows=True)
    assert len(games) == 3 and len(data) == 3
    assert parameter_hash(model) == actor_hash
    manifest = {"shards": [save_shard(tmp_path, 0, data)]}
    assert audit_dataset(tmp_path, manifest)["games"] == 3
    rows = load_rows(tmp_path, manifest)
    assert len(rows) == len(records)
    assert all(r['behavior_probabilities'][r['chosen']] == 1 for r in rows)
    for game in data:
        cash = next(g['cash'] for g in games if g['seed'] == game['seed'])
        assert all(r['terminal_cash'] == cash / 100 for r in game['rows'])
    optimizer = make_optimizer(model)
    rng = np.random.default_rng(13)
    updates, logs = update(model, optimizer, rows, rng, 0, batch_size=32)
    assert logs and updates == len(logs) and parameter_hash(model) != actor_hash
    assert all(l['cql_loss'] == l['auxiliary_loss'] == 0 for l in logs)
    assert parameter_hash(parent) == parent_hash
    assert all(int(v['step']) == updates for v in optimizer.state.values())
    again, _ = update(model, optimizer, rows, rng, updates, batch_size=32)
    assert all(int(v['step']) == again for v in optimizer.state.values())
    json.dumps(records)
