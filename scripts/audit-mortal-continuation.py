"""Recompute real sampling, single-root switching, matched rows and Adam/RNG chain."""

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from zimortal.training.active import atomic_json
from zimortal.training.mortal_boltzmann import sampling_probability
from zimortal.training.mortal_data import load_rows
from zimortal.training.mortal_diagnostics import distribution
from zimortal.training.mortal_online import learning_rate
from zimortal.training.online_rl import digest

parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=Path, default=Path('data/generated/mortal-continuation-v1'))
args = parser.parse_args()
root = args.directory
config = json.loads((root/'config.json').read_text())
if json.loads((root/'progress.json').read_text())['phase'] != 'complete_stop':
    raise RuntimeError('audit requires the completed fixed three-generation experiment')
for path, sha in config['frozen_sha256'].items():
    if digest(path) != sha:
        raise RuntimeError(f'frozen source/model changed: {path}')
official_path = root/'official-source'/'manifest.json'
assert digest(official_path) == config['official_manifest_sha256']
official = json.loads(official_path.read_text())
assert official['commit'] == '0cff2b52982be5b1163aa9a62fb01f03ce91e0d2'
for name, info in official['files'].items():
    candidates = [root/'official-source'/name.removeprefix('mortal/'), root/'official-source'/name.replace('/', '__')]
    file = next(p for p in candidates if p.exists())
    raw = file.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == info['sha256']
    assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == info['git_blob_sha1']
results, previous = [], {}
rngs = {a: np.random.default_rng(20261015) for a in ('A', 'B')}
updates = dict.fromkeys(rngs, 0)
for generation in range(1, 4):
    gen = root/f'gen-{generation:02d}'
    selection = json.loads((gen/'matched-root-selection.json').read_text())
    cohort = {tuple(k) for k in selection['common_seed_seat']}
    review, root_actions, prefixes = {}, {}, {}
    for arm in ('A', 'B'):
        armroot = gen/arm
        result = json.loads((armroot/'training-result.json').read_text())
        if generation > 1:
            assert result['producer_state_sha256'] == previous[arm]
        previous[arm] = result['updated_state_sha256']
        assert result['training_rows'] == len(cohort)
        rngs[arm].permutation(len(cohort))
        updates[arm] += result['updates_this_generation']
        opt = torch.load(armroot/'optimizer.pt', map_location='cpu', weights_only=False)
        assert opt['updates'] == updates[arm] == result['updates']
        assert opt['numpy_rng'] == rngs[arm].bit_generator.state
        assert opt['checkpoint_sha256'] == digest(armroot/'checkpoint.pt') == result['checkpoint_sha256']
        assert opt['metadata']['training_selection_sha256'] == digest(gen/'matched-root-selection.json')
        assert all(int(s['step']) == updates[arm] for s in opt['optimizer']['state'].values())
        assert all(g['lr'] == learning_rate(updates[arm]-1) for g in opt['optimizer']['param_groups'])
        logs = json.loads((armroot/'updates.json').read_text())
        assert sum(r['batch'] for r in logs) == len(cohort)
        assert all(r['cql_loss'] == r['auxiliary_loss'] == 0 for r in logs)
        roots, generators, steps = {}, {}, {}
        family_cash = defaultdict(list)
        count, switches, post = 0, 0, 0
        draws_by_family, roots_by_family = defaultdict(int), defaultdict(int)
        for path in sorted(armroot.glob('evidence-*/decisions-*.jsonl.gz')):
            with gzip.open(path, 'rt') as stream:
                for line in stream:
                    r = json.loads(line)
                    key = (r['seed'], r['seat'])
                    assert r['step'] > steps.get(key, -1)
                    steps[key] = r['step']
                    is_root = key not in roots and r['family'] != 'hu/pass'
                    assert r['is_training_root'] == is_root
                    greedy = r['family'] == 'hu/pass' or (arm == 'B' and key in roots)
                    q = np.asarray(r['q'], np.float32)
                    ref = np.asarray(r['probabilities'], np.float32)
                    score = q + np.float32(.3)*np.log(ref+np.float32(1e-8))
                    np.testing.assert_allclose(r['scores'], score, atol=1e-6)
                    p = sampling_probability(np.array(r['scores']), np.ones(len(q), bool), conservative_hu=greedy)
                    np.testing.assert_allclose(r['actor_probabilities'], p, atol=1e-12)
                    if greedy:
                        assert r['chosen'] == int(np.argmax(r['scores']))
                        assert r['actual_behavior_probability'] == 1
                    else:
                        if key not in generators:
                            generators[key] = np.random.default_rng(np.random.SeedSequence([*key, 20261016]))
                        assert r['chosen'] == int(generators[key].choice(len(p), p=p))
                    assert r['switch_trigger'] == (arm == 'B' and is_root)
                    if arm == 'B' and key in roots:
                        post += 1
                    if is_root:
                        roots[key] = (r['step']-1, r['chosen'])
                        if key in cohort:
                            family_cash[r['family']].append(r['terminal_cash'])
                    if r['execution_mode'] == 'boltzmann':
                        draws_by_family[r['family']] += 1
                    if is_root:
                        roots_by_family[r['family']] += 1
                    switches += int(r['switch_trigger'])
                    count += 1
        assert len(roots) == result['all_root_draws']
        assert switches == result['switch_triggers']
        manifest = json.loads((armroot/'data'/'manifest.json').read_text())
        rows = load_rows(armroot/'data', manifest)
        expected = {}
        for r in rows:
            key = (int(r['seed']), int(r['seat']))
            if r['family'] != 'hu/pass' and (key not in expected or r['history_step'] < expected[key][0]):
                expected[key] = (int(r['history_step']), int(r['chosen']))
        assert expected == roots
        root_actions[arm] = {k: roots[k] for k in cohort}
        if generation == 1:
            prefixes[arm] = {}
            for shard in manifest['shards']:
                with gzip.open(armroot/'data'/shard['replays'], 'rt') as stream:
                    games = json.load(stream)
                for game in games:
                    key = (game['seed'], game['seat'])
                    if key in cohort:
                        history = json.loads(game['state'])['history']
                        prefix = json.dumps(history[:roots[key][0]+1], sort_keys=True).encode()
                        prefixes[arm][key] = hashlib.sha256(prefix).hexdigest()
        review[arm] = {'decisions': count, 'root_draws': len(roots), 'matched_training_rows': len(cohort),
                       'switches': switches, 'verified_B_post_root_decisions': post, 'updates': updates[arm],
                       'Boltzmann_draws_by_family': dict(draws_by_family), 'root_draws_by_family': dict(roots_by_family),
                       'family_root_cash_yuan': {k: distribution(v) for k, v in family_cash.items()},
                       'actual_sampling_and_switching_verified': True, 'Adam_and_RNG_continuous': True}
        del rows
    if generation == 1:
        assert root_actions['A'] == root_actions['B']
        assert prefixes['A'] == prefixes['B'] and len(prefixes['A']) == len(cohort)
    assert updates['A'] == updates['B'] and rngs['A'].bit_generator.state == rngs['B'].bit_generator.state
    results.append({'generation': generation, 'arms': review})
lock = json.loads((root/'final-candidate-lock.json').read_text())
assert lock['generation'] == 3
assert all(lock[f'{a}_sha256'] == digest(root/f'gen-03/{a}/checkpoint.pt') for a in ('A', 'B'))
atomic_json(root/'independent-training-review.json', {'generations': results, 'frozen_sha_verified': True,
                                                     'fixed_Gen3_candidates_verified': True})
print(json.dumps(results, indent=2))
