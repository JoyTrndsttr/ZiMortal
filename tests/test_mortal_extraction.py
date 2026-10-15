import numpy as np
import pytest
import torch

from zimortal.training.mortal_extraction import extraction_scores
from zimortal.training.mortal_extraction_experiment import choose_development


def test_family_constraints_are_independent_and_do_not_modify_Q():
    q = torch.tensor([[1., 2., 100., 999.]]).repeat(4, 1)
    p = torch.tensor([[.8, .15, .05, 0.]]).repeat(4, 1)
    mask = torch.tensor([[True, True, True, False]]).repeat(4, 1)
    parent = torch.zeros(4, dtype=torch.long)
    families = torch.arange(4)
    original = q.clone()
    rule = {"mode": "family", "discard": True, "chi": False}
    scores, keep = extraction_scores(q, p, mask, parent, families, rule)
    assert scores.argmax(-1).tolist() == [1, 2, 2, 0]
    assert keep[:, 0].all()
    assert not keep[:, 3].any()
    assert torch.equal(q, original)
    rule.update(discard=False, chi=True)
    assert extraction_scores(q, p, mask, parent, families, rule)[0].argmax(-1).tolist() == [2, 1, 2, 0]


def test_anchor_is_exact_log_probability_score_without_Q_mutation():
    q = torch.tensor([[0., .2, .9]])
    p = torch.tensor([[.8, .19, .01]])
    mask = torch.ones_like(q, dtype=torch.bool)
    score, keep = extraction_scores(q, p, mask, torch.tensor([0]), torch.tensor([0]), {"mode": "anchor", "lambda": .3})
    assert torch.allclose(score, q + .3 * (p + 1e-8).log())
    assert score.argmax(-1).item() == 0
    assert keep.all()
    assert q[0, 2].item() == pytest.approx(.9)


def test_top_support_rescues_parent_and_respects_ragged_legal_mask():
    q = torch.tensor([[0., 10., 100., 999.], [0., 1., 2., 999.]])
    p = torch.tensor([[.05, .8, .15, 0.], [.9, .1, 0., 0.]])
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    score, keep = extraction_scores(q, p, mask, torch.tensor([0, 0]), torch.tensor([0, 1]), {"mode": "top", "discard_k": 2, "chi_k": 3})
    assert keep.tolist() == [[True, True, True, False], [True, True, False, False]]
    assert score.argmax(-1).tolist() == [2, 1]
    assert not torch.isnan(score).any()


def test_selection_uses_development_cash_only_with_deterministic_ties():
    dev = {n: [{"cash": c}] for n, c in {
        "C-discard": 1, "C-chi": 2, "C-both": 2,
        "D-lambda-0.01": 3, "D-lambda-0.3": 4,
        "E-top-2-2": 5, "E-top-3-3": 4,
    }.items()}
    assert choose_development(dev) == {"C-family": "C-both", "D-anchor": "D-lambda-0.3", "E-top": "E-top-2-2"}


def test_fixed_policy_evaluates_full_legal_Q_before_support_filter():
    from torch import nn

    from zimortal.engine import Action, ActionType
    from zimortal.model.encoding import encode_action
    from zimortal.training.mortal_extraction import ExtractionPolicy

    class Parent(nn.Module):
        def forward(self, x, a, m):
            return torch.tensor([[4., 2., -10.]]), torch.zeros(1), {}

    class FixedQ(nn.Module):
        def forward_q(self, x, a, m):
            assert m.all()
            return torch.tensor([[1., 2., 100.]]), torch.zeros(1)

    actions = [Action(ActionType.DISCARD, 0, tile=i) for i in range(3)]
    a = torch.from_numpy(np.stack([encode_action(v, 0) for v in actions]))[None]
    behavior = {"discard": 1.5, "chi/pass": 1.5, "peng/pass": 2.5, "hu/pass": 1., "hu_pass_probability": .02}
    policy = ExtractionPolicy(FixedQ(), Parent(), behavior, {"mode": "support"}).eval()
    scores = policy(torch.zeros(1, 55, 20), a, torch.ones(1, 3, dtype=torch.bool))[0]
    assert scores.argmax(-1).item() == 1
    assert policy.last[0][0, 2] == 100.


def test_complete_evaluation_preserves_frozen_parameters_and_buffers():
    import json

    from zimortal.model.mortal_faithful import MortalFaithfulQNet
    from zimortal.model.network import PolicyValueNet
    from zimortal.training.mortal_extraction import ExtractionPolicy, summarize_families
    from zimortal.training.online_rl import play

    torch.manual_seed(71)
    parent = PolicyValueNet(feature_version="huxi").eval().requires_grad_(False)
    q = MortalFaithfulQNet(width=16, blocks=1).eval().requires_grad_(False)
    before = {k: v.clone() for k, v in q.state_dict().items()}
    behavior = {"discard": 1.5, "chi/pass": 1.5, "peng/pass": 2.5, "hu/pass": 1., "hu_pass_probability": .02}
    policy = ExtractionPolicy(q, parent, behavior, {"mode": "anchor", "lambda": .1}).eval()
    records = []
    rows, games, failures = play(policy, [parent], [(1130000000, 0, 0, 0), (1130000001, 1, 1, 0)],
                                decision_recorder=policy.record, decision_sink=records.extend)
    assert not failures and len(games) == 2 and records
    assert not rows
    assert all(torch.equal(before[k], v) for k, v in q.state_dict().items())
    assert all(p.grad is None for p in q.parameters())
    json.dumps(summarize_families(records))
    json.dumps(records)
