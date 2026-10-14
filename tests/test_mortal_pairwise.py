import numpy as np
import torch

from zimortal.training.mortal_pairwise import auxiliary_loss, confidence_weight


def test_uncertain_rollout_has_no_training_weight():
    assert (
        confidence_weight(
            {"paired_bootstrap_95": [-1, 9], "paired_delta": 4, "paired_standard_error": 2}
        )
        == 0
    )
    assert (
        confidence_weight(
            {"paired_bootstrap_95": [10, 30], "paired_delta": 20, "paired_standard_error": 2}
        )
        > 0.7
    )


def test_pairwise_gradient_preserves_rollout_direction():
    class Model:
        def __init__(self):
            self.q = torch.zeros(2, requires_grad=True)

        def forward_q(self, x, a, mask):
            return self.q.expand(len(x), 2), None

    model = Model()
    row = {
        "features": np.zeros((55, 20), np.float32),
        "encoded_actions": np.zeros((2, 3), np.float32),
        "parent": 0,
        "chosen": 1,
        "weight": 1,
        "stats": {"paired_delta": -20},
    }
    loss = auxiliary_loss(model, [row], "cpu")
    loss.backward()
    assert model.q.grad[0] < 0 < model.q.grad[1]
    row["weight"] = 0
    assert auxiliary_loss(model, [row], "cpu").item() == 0


def test_seed_cluster_accuracy_comparison_and_pair_alignment():
    from zimortal.training.mortal_pairwise_review import accuracy_gain

    a = {"rollout_delta": [10, 0, -5], "correct": [False, False, True], "seeds": [1, 1, 2]}
    b = {**a, "correct": [True, True, True]}
    result = accuracy_gain(a, b)
    assert result["gain"] == 0.5
    assert result["ci95"][0] >= 0
