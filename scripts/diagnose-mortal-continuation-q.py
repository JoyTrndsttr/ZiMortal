"""Read-only value/centered-advantage decomposition on existing matched A roots."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from zimortal.training.active import atomic_json
from zimortal.training.mortal_data import load_rows
from zimortal.training.mortal_online_long import q_probe
from zimortal.training.mortal_sarsa import parameter_hash
from zimortal.training.runtime import load_model


def components(before, after):
    total = value = advantage = 0.
    candidates = changed = 0
    for a, b in zip(before, after, strict=True):
        delta = b.astype(float)-a.astype(float)
        mean = delta.mean()
        total += float(np.sum(delta**2))
        value += len(delta)*mean**2
        advantage += float(np.sum((delta-mean)**2))
        candidates += len(delta)
        changed += int(a.argmax() != b.argmax())
    if not np.isclose(total, value+advantage, atol=1e-10):
        raise RuntimeError('Q decomposition does not sum')
    return {'rows': len(before), 'legal_candidates': candidates,
            'total_q_rmse': float(np.sqrt(total/candidates)),
            'state_common_value_rmse': float(np.sqrt(value/candidates)),
            'centered_advantage_rmse': float(np.sqrt(advantage/candidates)),
            'value_squared_drift_fraction': value/total if total else None,
            'raw_argmax_disagreement': changed/max(1, len(before))}


parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=Path, default=Path('data/generated/mortal-continuation-v1'))
parser.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
args = parser.parse_args()
root = args.directory
torch.set_num_threads(2)
if json.loads((root/'progress.json').read_text())['phase'] != 'complete_stop':
    raise RuntimeError('diagnostics only after the fixed experiment has completed')
config = json.loads((root/'config.json').read_text())
results = []
for generation in range(1, 4):
    gen = root/f'gen-{generation:02d}'
    selection = json.loads((gen/'matched-root-selection.json').read_text())
    keys = {tuple(k) for k in selection['common_seed_seat']}
    all_rows = load_rows(gen/'A/data', json.loads((gen/'A/data/manifest.json').read_text()))
    index = {}
    for row in all_rows:
        key = (int(row['seed']), int(row['seat']))
        if key in keys and row['family'] != 'hu/pass' and (key not in index or row['history_step'] < index[key]['history_step']):
            index[key] = row
    rows = [index[k] for k in sorted(keys)]
    outputs = {}
    for arm in ('A', 'B'):
        for stage in ('producer', 'updated'):
            path = (config['initial_checkpoint'] if generation == 1 else root/f'gen-{generation-1:02d}'/arm/'checkpoint.pt') if stage == 'producer' else gen/arm/'checkpoint.pt'
            model = load_model(path, args.device).requires_grad_(False)
            before_hash = parameter_hash(model)
            outputs[arm, stage] = q_probe(model, rows, device=args.device)
            if parameter_hash(model) != before_hash:
                raise RuntimeError('diagnostic inference mutated weights/BN')
            del model
    results.append({'generation': generation, 'common_probe': 'all matched A training roots, no outcome filtering',
                    'A_update': components(outputs['A', 'producer'], outputs['A', 'updated']),
                    'B_update': components(outputs['B', 'producer'], outputs['B', 'updated']),
                    'B_minus_A_updated': components(outputs['A', 'updated'], outputs['B', 'updated'])})
atomic_json(root/'q-output-components.json', {'device': args.device, 'not_for_training': True, 'generations': results})
print(json.dumps(results, indent=2))
