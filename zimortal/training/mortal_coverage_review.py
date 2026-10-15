"""Post-lock, read-only behavior coverage analysis; no training or selection."""

import gzip
import json
from pathlib import Path

import numpy as np
import torch

from zimortal.training.active import atomic_json
from zimortal.training.mortal_coverage_sampling import enhance_probability
from zimortal.training.mortal_data import load_rows
from zimortal.training.mortal_faithful_diagnostics import raw_diagnostics
from zimortal.training.online_rl import digest, pack, paired_summary
from zimortal.training.runtime import load_model


def first_decision_comparison(groups):
    first = {}
    for name, rows in groups.items():
        indexed = {}
        for r in rows:
            seed = int(r["seed"])
            if seed not in indexed or r["history_step"] < indexed[seed]["history_step"]:
                indexed[seed] = r
        first[name] = indexed
    left, right = first["A-original"], first["B-coverage"]
    families = {}
    unmatched = 0
    for seed in left.keys() & right.keys():
        a, b = left[seed], right[seed]
        same = np.array_equal(a["features"], b["features"]) and np.array_equal(a["actions"], b["actions"])
        if not same:
            unmatched += 1
            continue
        f = str(a["family"])
        d = families.setdefault(f, {"matched_roots": 0, "A_non_greedy": 0, "B_non_greedy": 0})
        d["matched_roots"] += 1
        d["A_non_greedy"] += int(a["chosen"] != a["parent_greedy"])
        d["B_non_greedy"] += int(b["chosen"] != b["parent_greedy"])
    return {"families": families, "unmatched_first_states": unmatched,
            "note": "paired first learner decisions only; not repeated sampling of every visited state"}


def test_override_records(root, name, rows):
    model = load_model(root / name / "fit" / "selected-common.pt", "cpu")
    records = []
    config = json.loads((root / 'config.json').read_text())['arms'][name]
    with torch.inference_mode():
        for start in range(0, len(rows), 256):
            batch = rows[start:start + 256]
            q = model.forward_q(*pack([r["features"] for r in batch], [r["actions"] for r in batch], "cpu"))[0].numpy()
            for r, values in zip(batch, q, strict=True):
                parent = int(r["parent_greedy"])
                chosen = int(values[:len(r["actions"])].argmax())
                p = float(r["reference_probabilities"][chosen])
                own_p = float(enhance_probability(r["reference_probabilities"], parent, str(r["family"]), config)[chosen])
                records.append({"seed": int(r["seed"]), "family": str(r["family"]),
                                "changed": chosen != parent, "low": chosen != parent and p <= .1,
                                "probability": p, "training_behavior_probability": own_p,
                                "chosen": chosen, "parent": parent})
    return records


def paired_low_support(records_a, records_b):
    seeds = sorted({r["seed"] for r in records_a})
    def counts(rows):
        counts = {s: [0, 0] for s in seeds}
        for r in rows:
            counts[r["seed"]][0] += int(r["low"])
            counts[r["seed"]][1] += int(r["changed"])
        return np.array([counts[s] for s in seeds])
    a, b = counts(records_a), counts(records_b)
    totals = {"A_overrides": int(a[:, 1].sum()), "B_overrides": int(b[:, 1].sum())}
    if not totals["A_overrides"] or not totals["B_overrides"]:
        return {**totals, "A_fraction": None, "B_fraction": None,
                "B_minus_A_seed_bootstrap_95": None}
    rng = np.random.default_rng(1414)
    indices = rng.integers(len(seeds), size=(10000, len(seeds)))
    aa, bb = a[indices].sum(1), b[indices].sum(1)
    delta = bb[:, 0] / np.maximum(1, bb[:, 1]) - aa[:, 0] / np.maximum(1, aa[:, 1])
    return {**totals, "A_fraction": float(a[:, 0].sum() / max(1, a[:, 1].sum())),
            "B_fraction": float(b[:, 0].sum() / max(1, b[:, 1].sum())),
            "B_minus_A_seed_bootstrap_95": np.quantile(delta, [.025, .975]).tolist()}


def run():
    torch.set_num_threads(2)
    root = Path('data/generated/mortal-coverage-v1')
    result = json.loads((root / 'result.json').read_text())
    lock = json.loads((root / 'selection-lock.json').read_text())
    common = lock['common_update']
    for name, sha in lock['checkpoints'].items():
        fit = root / name / 'fit'
        if digest(fit / 'selected-common.pt') != sha:
            raise RuntimeError('selected checkpoint changed')
        opt = torch.load(fit / 'selected-common-optimizer.pt', map_location='cpu', weights_only=True)
        steps = {int(v['step']) for v in opt['optimizer']['state'].values()}
        if steps != {common} or opt['updates'] != common or opt['checkpoint_sha256'] != sha:
            raise RuntimeError('paired optimizer step mismatch')
    groups = {n: load_rows(root / n / 'data', json.loads((root / n / 'data' / 'manifest.json').read_text()))
              for n in ('A-original', 'B-coverage')}
    data = root / 'heldout' / 'data'
    heldout = load_rows(data, json.loads((data / 'manifest.json').read_text()))
    test = [r for r in heldout if not int(r['seed']) % 2]
    records = {n: test_override_records(root, n, test) for n in groups}
    with gzip.open(root / 'postlock-test-override-records.jsonl.gz', 'wt') as file:
        for n, rows in records.items():
            for row in rows:
                file.write(json.dumps({'arm': n, **row}) + '\n')
    low = {}
    for family in ('all', 'discard', 'chi/pass', 'peng/pass', 'hu/pass'):
        a, b = ([r for r in records[n] if family == 'all' or r['family'] == family] for n in groups)
        if not a:
            continue
        low[family] = paired_low_support(a, b)
    outcomes = json.loads((root / 'screening' / 'screening-outcomes.json').read_text())
    raw_direct = result['B_minus_A']['raw']['0']
    guarded_direct = result['B_minus_A']['guarded']['0']
    differences = {}
    for n in groups:
        raw = {(r['seed'], r['seat'], r['opponent']): r for r in outcomes[n + '-raw']}
        differences[n] = [{**r, 'cash': r['cash'] - raw[(r['seed'], r['seat'], r['opponent'])]['cash']}
                          for r in outcomes[n + '-guarded']]
    reliance = paired_summary(differences['B-coverage'], differences['A-original'], 1414)
    primary_low = {n: result['offline'][n]['raw']['overrides']['low_support_fraction']['0.1'] for n in groups}
    recommendation = (
        raw_direct['paired_seed_bootstrap_95'][0] > 0
        and primary_low['B-coverage'] < primary_low['A-original']
    )
    own_low = {n: sum(r['changed'] and r['training_behavior_probability'] <= .1 for r in rr) / max(1, sum(r['changed'] for r in rr)) for n, rr in records.items()}
    absolute_low = {n: result['offline'][n]['raw']['changed_action_rate'] * primary_low[n] for n in groups}
    review = {'common_selected_update': common, 'primary_cuda_test_low_support': primary_low,
              'own_training_behavior_test_low_support_cpu_diagnostic': own_low,
              'primary_low_override_per_all_test_decisions': absolute_low,
              'first_decision_comparison': first_decision_comparison(groups),
              'test_reference_low_support_cpu_diagnostic': low,
              'cash_raw_B_minus_A': raw_direct,
              'cash_guarded_B_minus_A': guarded_direct,
              'guard_reliance_B_minus_A': reliance,
              'raw_trajectory_diagnostics': {n: raw_diagnostics(root / 'screening' / f'decisions-{n}-raw.jsonl.gz') for n in groups},
              'coverage_direction_supported_in_this_pilot': recommendation,
              'additional_exploration_started': False, 'online_started': False,
              'note': 'CPU test argmax diagnostics may differ at numeric ties from primary CUDA metrics; original support reference throughout'}
    atomic_json(root / 'review.json', review)
    print(json.dumps({k: v for k, v in review.items() if k != 'raw_trajectory_diagnostics'}, indent=2))


if __name__ == '__main__':
    run()
