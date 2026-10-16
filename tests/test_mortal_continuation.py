import numpy as np
import torch
from test_mortal_deviation import BEHAVIOR, FixedQ, Parent

from zimortal.engine import Action, ActionType
from zimortal.model.encoding import encode_action
from zimortal.training.mortal_continuation import ContinuationPolicy, matched_roots
from zimortal.training.mortal_data import audit_dataset, load_rows, save_shard
from zimortal.training.mortal_online import collect
from zimortal.training.mortal_sarsa import parameter_hash


def test_real_single_root_then_frozen_greedy_and_shared_initial_root(tmp_path):
    actors = {arm: ContinuationPolicy(FixedQ(), Parent(), BEHAVIOR, greedy_continuation=arm == 'B') for arm in ('A', 'B')}
    specs = [(1249999900+i, i%3, i%3, 0) for i in range(8)]
    datasets, evidence = {}, {}
    for arm, actor in actors.items():
        before = parameter_hash(actor.q)
        _, records, data = collect(actor, actor.parent, specs, tmp_path/arm, arm, training_rows=True)
        assert parameter_hash(actor.q) == before
        manifest = {'shards': [save_shard(tmp_path/arm, 0, data)]}
        assert audit_dataset(tmp_path/arm, manifest)['games'] == 8
        datasets[arm] = load_rows(tmp_path/arm, manifest)
        evidence[arm] = records
        for seed, _, seat, _ in specs:
            rows = [r for r in records if (r['seed'], r['seat']) == (seed, seat)]
            roots = [r for r in rows if r['is_training_root']]
            assert len(roots) <= 1
            if not roots:
                assert all(r['family'] == 'hu/pass' for r in rows)
                continue
            root = roots[0]
            assert root['family'] != 'hu/pass'
            assert root['switch_trigger'] == (arm == 'B')
            for r in rows:
                if arm == 'B' and r['step'] > root['step']:
                    assert r['chosen'] == r['anchored_greedy']
                    assert r['actual_behavior_probability'] == 1
                    assert not r['switch_trigger']
    selected, m = matched_roots(datasets['A'], datasets['B'])
    assert len(m['common_seed_seat']) == len(selected['A']) == len(selected['B'])
    for a, b in zip(selected['A'], selected['B'], strict=True):
        assert np.array_equal(a['features'], b['features'])
        assert np.array_equal(a['actions'], b['actions'])
        assert a['chosen'] == b['chosen']
    assert any(r['actual_behavior_probability'] < 1 for r in evidence['A'] if not r['is_training_root'] and r['family'] != 'hu/pass')


def test_hu_does_not_trigger_root_and_forward_alone_does_not_commit():
    actor = ContinuationPolicy(FixedQ(), Parent(), BEHAVIOR, greedy_continuation=True)
    hu = [Action(ActionType.HU, 0), Action(ActionType.PASS, 0)]
    actions = torch.from_numpy(np.stack([encode_action(a, 0) for a in hu]))[None]
    actor.prepare([(1, 0, 0, 0)])
    actor(torch.zeros(1, 55, 20), actions, torch.ones(1, 2, dtype=torch.bool))
    assert actor.root_flags == [False] and not actor.root_steps


def test_root_matching_is_chronological_outcome_blind_and_owns_arrays():
    def row(seed, step, family, cash):
        return {'seed': seed, 'seat': 0, 'history_step': step, 'family': family, 'terminal_cash': cash, 'features': np.zeros((55, 20))}
    a = [row(1, 8, 'discard', 100), row(1, 2, 'chi/pass', 100), row(2, 1, 'discard', -100)]
    b = [row(1, 0, 'hu/pass', -100), row(1, 2, 'chi/pass', -100)]
    selected, m = matched_roots(a, b)
    assert m['common_seed_seat'] == [(1, 0)]
    assert m['unmatched_A_only'] == [(2, 0)]
    assert selected['A'][0]['history_step'] == 2
    assert selected['A'][0]['terminal_cash'] == 100 and selected['B'][0]['terminal_cash'] == -100
    assert not np.shares_memory(selected['A'][0]['features'], a[1]['features'])


def test_continuation_protocol_disjoint_seeds_and_fixed_holdout():
    from zimortal.training.mortal_continuation_experiment import seed_specs

    seen = set()
    for generation in range(1, 4):
        for evaluation in (False, True):
            specs = seed_specs(generation, evaluation=evaluation)
            seeds = {s[0] for s in specs}
            assert not seen & seeds
            seen |= seeds
            assert len(specs) == (3000 if evaluation else 5000)
            if evaluation:
                assert len(seeds) == 1000
    holdout = seed_specs(holdout=True)
    assert len(holdout) == 9000
    assert not seen & {s[0] for s in holdout}
    assert len({s[0] for s in holdout}) == 3000
