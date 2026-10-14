import numpy as np
import torch

from zimortal.engine import Action, ActionType
from zimortal.model.encoding import encode_action
from zimortal.training.mortal_data import family_exploration
from zimortal.training.mortal_support import behavior_probability, support_mask

CONFIG = {
    "discard": 1.5,
    "chi/pass": 1.5,
    "peng/pass": 2.5,
    "hu/pass": 1.0,
    "hu_pass_probability": 0.02,
}


def test_support_strict_parent_rescue_padding():
    p = torch.tensor([[0.01, 0.03, 0.05, 0.91, 0.0]])
    m = torch.tensor([[True, True, True, True, False]])
    parent = torch.tensor([0])
    assert support_mask(p, m, parent, 0.03).tolist() == [[True, False, True, True, False]]
    assert support_mask(p, m, parent, 1.0).tolist() == [[True, False, False, False, False]]
    assert torch.equal(support_mask(p, m, parent, None), m)


def test_visible_behavior_matches_original_temperature_and_hu_override():
    for kind in (ActionType.DISCARD, ActionType.CHI, ActionType.PENG, ActionType.HU):
        actions = [Action(kind, 0, tile=1), Action(ActionType.PASS, 0, tile=1)]
        encoded = torch.from_numpy(np.stack([encode_action(a, 0) for a in actions]))[None]
        logits = torch.tensor([[2.0, -1.0]])
        actual = behavior_probability(logits, encoded, torch.ones(1, 2, dtype=torch.bool), CONFIG)[
            0
        ].numpy()
        expected = family_exploration([2.0, -1.0], actions, CONFIG)
        np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_support_scores_full_dueling_mask_before_filtering():
    from torch import nn

    from zimortal.training.mortal_support import SupportPolicy

    actions = [Action(ActionType.DISCARD, 0, tile=i) for i in range(3)]
    encoded = torch.from_numpy(np.stack([encode_action(a, 0) for a in actions]))[None]
    mask = torch.ones(1, 3, dtype=torch.bool)

    class Parent(nn.Module):
        def forward(self, x, a, m):
            return torch.tensor([[4.0, 2.0, -10.0]]), torch.zeros(1), {}

    class Q(nn.Module):
        def forward_q(self, x, a, m):
            assert torch.equal(m, mask)
            return torch.tensor([[1.0, 2.0, 100.0]]), torch.zeros(1)

    policy = SupportPolicy(Q(), Parent(), CONFIG, 0.05)
    scores = policy(torch.zeros(1, 55, 20), encoded, mask)[0]
    assert scores.argmax(-1).item() == 1
    assert torch.isneginf(scores[0, 2])
    assert policy.last[0][0, 2] == 100.0
    assert scores[0, 0].item() == 1.0


def test_deferred_heldout_never_calls_test_regression(monkeypatch, tmp_path):
    import zimortal.training.mortal_offline as module
    from zimortal.model.encoding import ACTION_DIM
    from zimortal.model.mortal_q import MortalQNet

    row = {
        "features": np.zeros((55, 20), np.float32),
        "actions": np.zeros((2, ACTION_DIM), np.float32),
        "chosen": 0,
        "terminal_cash": 0.0,
        "steps_to_terminal": 0,
    }
    sentinel = [{"test_must_not_be_read": True}]

    def regression(model, rows, **kwargs):
        assert rows is not sentinel
        return {"rmse": 1.0}

    monkeypatch.setattr(module, "regression", regression)
    result = module.fit(
        MortalQNet(width=8, latent=32, blocks=1),
        [row],
        [row],
        sentinel,
        tmp_path / "fit",
        epochs=1,
        evaluate_test=False,
    )
    assert result["heldout"] is None


def test_complete_decision_sink_uses_actual_game_cash():
    from zimortal.model.mortal_q import MortalQNet
    from zimortal.model.network import PolicyValueNet
    from zimortal.training.mortal_support import SupportPolicy
    from zimortal.training.online_rl import play

    parent = PolicyValueNet(width=8, feature_version="huxi").eval()
    policy = SupportPolicy(
        MortalQNet(width=8, latent=32, blocks=1).eval(), parent, CONFIG, 0.1
    ).eval()
    evidence = []
    _, games, errors = play(
        policy,
        [parent],
        [(118, 1, 0, 0), (119, 2, 1, 0)],
        slots=2,
        decision_recorder=policy.record,
        decision_sink=evidence.extend,
    )
    assert not errors and len(games) == 2 and evidence
    cash = {(r["seed"], r["seat"]): r["cash"] for r in games}
    for r in evidence:
        assert r["terminal_cash"] == cash[(r["seed"], r["seat"])]
        assert r["changed"] == (r["chosen"] != r["parent"])
        assert not r["changed"] or r["behavior_probability"] > 0.1
        assert len(r["actions"]) == r["candidate_count"]


def test_existing_experiment_cannot_overwrite_progress(monkeypatch, tmp_path):
    import sys

    import pytest

    from zimortal.training.mortal_support_experiment import main

    progress = tmp_path / "progress.json"
    progress.write_text('{"status":"complete"}')
    monkeypatch.setattr(sys, "argv", ["experiment", "--directory", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert progress.read_text() == '{"status":"complete"}'
