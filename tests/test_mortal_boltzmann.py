import json

import numpy as np
import pytest
import torch
from test_mortal_deviation import BEHAVIOR, FixedQ, Parent

from zimortal.engine import Action, ActionType
from zimortal.model.encoding import encode_action
from zimortal.training.mortal_boltzmann import AnchoredBoltzmann, sampling_probability
from zimortal.training.mortal_boltzmann_experiment import seed_specs
from zimortal.training.mortal_data import audit_dataset, load_rows, save_shard
from zimortal.training.mortal_online import collect
from zimortal.training.mortal_sarsa import parameter_hash


def test_sampling_exact_anchored_temperature_padding_and_conservative_hu():
    s = np.array([0., 1., -np.inf])
    p = sampling_probability(s, [True, True, False])
    assert p == pytest.approx([1/(1+np.exp(2)), 1/(1+np.exp(-2)), 0])
    assert sampling_probability(s, [True, True, False], conservative_hu=True).tolist() == [0, 1, 0]
    with pytest.raises(ValueError):
        sampling_probability(s, [False, False, False])


def test_per_game_rng_is_independent_of_batch_slot_and_hu_preserves_anchor():
    actions = [Action(ActionType.DISCARD, 0, tile=i) for i in range(2)]
    encoded = torch.from_numpy(np.stack([encode_action(a, 0) for a in actions]))
    p, other = [AnchoredBoltzmann(FixedQ(), Parent(), BEHAVIOR) for _ in range(2)]
    for _ in range(5):
        p.prepare([(1, 0, 0, 0), (2, 0, 0, 0)])
        out = p(torch.zeros(2, 55, 20), encoded[None].repeat(2, 1, 1), torch.ones(2, 2, dtype=torch.bool))[0]
        other.prepare([(2, 0, 0, 0)])
        independent = other(torch.zeros(1, 55, 20), encoded[None], torch.ones(1, 2, dtype=torch.bool))[0]
        assert out.argmax(-1)[1] == independent.argmax(-1)[0]
        np.testing.assert_allclose(p.last[0][0], [0., 1.])
        assert np.isclose(p.last[1][0].sum(), 1)
    hu = [Action(ActionType.HU, 0), Action(ActionType.PASS, 0)]
    ha = torch.from_numpy(np.stack([encode_action(a, 0) for a in hu]))[None]
    p.prepare([(3, 0, 0, 0)])
    out = p(torch.zeros(1, 55, 20), ha, torch.ones(1, 2, dtype=torch.bool))[0]
    assert out.argmax(-1).item() == int(p.last[-1][0].argmax())
    assert set(p.actor_probabilities[0]) <= {0., 1.}


def test_real_sampling_trajectory_has_actual_probabilities_and_replays(tmp_path):
    p = AnchoredBoltzmann(FixedQ(), Parent(), BEHAVIOR).eval()
    q_hash, parent_hash = parameter_hash(p.q), parameter_hash(p.parent)
    _, records, games = collect(p, p.parent, [(1219999900+i, i%3, i%3, 0) for i in range(3)],
                               tmp_path, 'sampled', training_rows=True)
    manifest = {'shards': [save_shard(tmp_path, 0, games)]}
    assert audit_dataset(tmp_path, manifest)['games'] == 3
    rows = load_rows(tmp_path, manifest)
    assert any(0 < r['behavior_probabilities'][r['chosen']] < 1 for r in rows)
    assert all(r['behavior_logprob'] == pytest.approx(np.log(r['behavior_probabilities'][r['chosen']])) for r in rows)
    assert parameter_hash(p.q) == q_hash and parameter_hash(p.parent) == parent_hash
    assert all(r['sampling_temperature'] == .5 for r in records)
    json.dumps(records)


def test_new_common_training_and_independent_evaluation_seeds():
    seen = set()
    for generation in range(1, 4):
        train, evaluation = seed_specs(generation), seed_specs(generation, True)
        assert len(train) == 3000 and len(evaluation) == 1500
        a, b = {s[0] for s in train}, {s[0] for s in evaluation}
        assert seen.isdisjoint(a | b) and a.isdisjoint(b)
        assert len(b) == 500
        seen |= a | b
