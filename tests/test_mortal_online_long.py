import numpy as np
import pytest
import torch

from zimortal.training.mortal_online import learning_rate
from zimortal.training.mortal_online_long import parameter_change, q_drift, safety_stop, seed_specs


def test_long_protocol_seeds_have_no_overlap_and_three_evaluation_seats():
    seen = set()
    for generation in range(1, 11):
        for evaluation in (False, True):
            specs = seed_specs(generation, evaluation=evaluation)
            seeds = {s[0] for s in specs}
            assert not seen & seeds
            seen |= seeds
            assert len(specs) == (3000 if evaluation else 5000)
            assert len(seeds) == (1000 if evaluation else 5000)
    holdout = seed_specs(holdout=True)
    assert len(holdout) == 15000 and not seen & {s[0] for s in holdout}
    with pytest.raises(ValueError):
        seed_specs(10, holdout=True)


def test_stop_requires_two_consecutive_extreme_upper_bounds():
    def r(upper):
        return {'cash': {'anchor': {'seed_bootstrap_95': [-10, upper]}}}
    assert not safety_stop([r(-6)])
    assert not safety_stop([r(-6), r(-5)])
    assert not safety_stop([r(-6), r(0), r(-6)])
    assert safety_stop([r(0), r(-6), r(-7)])
    assert learning_rate(320) == pytest.approx(3e-6)
    assert learning_rate(2000) == pytest.approx(learning_rate(320))


def test_drift_uses_same_rows_and_does_not_mutate_model():
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    with torch.no_grad():
        model.weight.add_(1)
    result = parameter_change(before, model)
    assert result['delta_l2'] == pytest.approx(2**.5)
    assert result['relative_delta_l2'] == pytest.approx(1)
    assert torch.equal(model.weight, torch.tensor([[2., 2.]]))
    result = q_drift([np.array([0., 1.])], [np.array([2., 1.])], [{'chosen': 0}])
    assert result['legal_q_rmse'] == pytest.approx(2**.5)
    assert result['chosen_q_delta']['mean'] == 2
    assert result['raw_argmax_change_rate'] == 1
    with pytest.raises(ValueError):
        q_drift([], [np.array([1.])], [])
