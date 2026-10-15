"""Audit the completed fixed 50k run: source hashes, Adam/RNG chain and actual sampling."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from zimortal.training.mortal_boltzmann import sampling_probability
from zimortal.training.mortal_online import learning_rate

parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=Path, default=Path('data/generated/mortal-online-long-v1'))
root = parser.parse_args().directory
paths = sorted(root.glob('gen-*/training-result.json'))
if len(paths) != 10:
    raise RuntimeError('audit requires all ten completed training generations')
config = json.loads((root / 'config.json').read_text())
for p, sha in config['frozen_sha256'].items():
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha, p
rng = np.random.default_rng(20261015)
updates = 0
results = []
previous = None
for path in paths:
    result = json.loads(path.read_text())
    gen = path.parent
    if previous is not None:
        assert result['producer_state_sha256'] == previous
    previous = result['updated_state_sha256']
    assert result['updates_this_generation'] == math.ceil(result['decisions']/256)
    updates += result['updates_this_generation']
    assert result['updates'] == updates
    rng.permutation(result['decisions'])
    opt = torch.load(gen / 'optimizer.pt', map_location='cpu', weights_only=False)
    assert opt['updates'] == updates
    assert opt['numpy_rng'] == rng.bit_generator.state
    assert all(g['lr'] == learning_rate(updates-1) for g in opt['optimizer']['param_groups'])
    assert all(int(s['step']) == updates for s in opt['optimizer']['state'].values())
    count = 0
    for file in sorted(gen.glob('evidence-*/decisions-*.jsonl.gz')):
        with gzip.open(file, 'rt') as stream:
            for line in stream:
                r = json.loads(line)
                q, ref = np.asarray(r['q'], np.float32), np.asarray(r['probabilities'], np.float32)
                score = q + np.float32(.3) * np.log(ref+np.float32(1e-8))
                assert np.allclose(r['scores'], score, atol=1e-6)
                p = sampling_probability(np.array(r['scores']), np.ones(len(q), bool), conservative_hu=r['family']=='hu/pass')
                np.testing.assert_allclose(r['actor_probabilities'], p, atol=1e-12)
                assert np.isclose(r['actual_behavior_probability'], p[r['chosen']])
                count += 1
    assert count == result['decisions']
    results.append({'generation': int(gen.name[-2:]), 'decisions': count, 'updates': updates,
                    'optimizer_continuous': True, 'rng_continuous': True, 'sampling_recomputed': True})
print(json.dumps(results, indent=2))
(root / 'independent-training-review.json').write_text(json.dumps(results, indent=2)+'\n')
