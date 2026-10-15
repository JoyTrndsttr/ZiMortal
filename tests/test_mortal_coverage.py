import numpy as np
import pytest

from zimortal.engine import Action
from zimortal.engine.types import ActionType
from zimortal.training.mortal_coverage_sampling import enhance_probability
from zimortal.training.mortal_data import family_exploration

CONFIG = {"coverage": {"mixture": .5, "minimum": .03, "alternatives": 3}}


def test_bounded_coverage_increases_supported_alternatives_only():
    p = np.array([.7, .15, .1, .04, .01])
    b = enhance_probability(p, 0, "discard", CONFIG)
    assert b.sum() == pytest.approx(1)
    assert b[0] == pytest.approx(.35)
    assert all(b[i] > p[i] for i in (1, 2, 3))
    assert b[4] == pytest.approx(.005)
    assert np.array_equal(enhance_probability([.99, .01], 0, "discard", CONFIG), [.99, .01])


def test_peng_and_hu_exploration_are_exactly_unchanged():
    p = np.array([.98, .02])
    for family in ("hu/pass", "peng/pass"):
        assert np.array_equal(enhance_probability(p, 0, family, CONFIG), p)
    base = {"discard": 1.5, "chi/pass": 1.5, "peng/pass": 2.5,
            "hu/pass": 1., "hu_pass_probability": .02}
    for kind in (ActionType.HU, ActionType.PENG):
        actions = [Action(kind, 0), Action(ActionType.PASS, 0)]
        assert np.array_equal(family_exploration([2., 0.], actions, base),
                              family_exploration([2., 0.], actions, base | CONFIG))


def test_reference_probabilities_roundtrip_and_complete_replay(tmp_path):
    import torch

    from zimortal.model.network import PolicyValueNet
    from zimortal.training.mortal_data import (
        audit_dataset,
        generate_games,
        load_rows,
        save_shard,
        specs,
    )

    torch.manual_seed(8)
    model = PolicyValueNet(feature_version="huxi").eval()
    config = {"discard": 1.5, "chi/pass": 1.5, "peng/pass": 2.5,
              "hu/pass": 1., "hu_pass_probability": .02, "record_reference": True} | CONFIG
    games, failures, _ = generate_games(model, specs(1100000000, 3), family_config=config)
    assert not failures and len(games) == 3
    shard = save_shard(tmp_path, 0, games)
    rows = load_rows(tmp_path, {"shards": [shard]})
    assert rows
    from zimortal.training.mortal_coverage_experiment import verify_reference_and_transform
    assert verify_reference_and_transform(rows, config)["maximum_transform_error"] < 1e-6
    assert audit_dataset(tmp_path, {"shards": [shard]})["games"] == 3


def test_executed_coverage_exports_numpy_decisions_as_json():
    import json

    from zimortal.training.mortal_coverage_experiment import coverage_evidence

    rows = [{"family": "discard", "chosen_kind": "discard", "chosen": np.int64(1),
             "parent_greedy": np.int64(0), "features": np.zeros((55, 20), dtype=np.float32),
             "actions": np.zeros((2, 3), dtype=np.float32),
             "reference_probabilities": np.array([.95, .05], dtype=np.float32),
             "behavior_probabilities": np.array([.475, .525], dtype=np.float32)}]
    stats = json.loads(json.dumps(coverage_evidence(rows)))
    assert stats["discard"]["original_low_support_executed"] == 1
    assert stats["discard"]["non_greedy_executed"] == 1
