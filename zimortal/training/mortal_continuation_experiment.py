"""Fixed three-generation MC continuation comparison; final candidate is Gen3."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .active import atomic_json
from .mortal_boltzmann_experiment import behavior_summary
from .mortal_continuation import ContinuationPolicy, matched_roots
from .mortal_credit import audit_screening
from .mortal_data import audit_dataset, load_rows, save_shard
from .mortal_deviation_stats import cluster_cohort
from .mortal_diagnostics import distribution
from .mortal_extraction import ExtractionPolicy
from .mortal_extraction_experiment import CHECKPOINT
from .mortal_offline import BASELINE, regression
from .mortal_online import checkpoint, collect, make_optimizer, update
from .mortal_online_experiment import assert_frozen
from .mortal_online_long import parameter_change, q_drift, q_probe
from .mortal_sarsa import parameter_hash
from .online_rl import digest, source_hashes
from .runtime import load_model

GENERATIONS = 3
GAMES = 5000


def seed_specs(generation=None, *, evaluation=False, holdout=False):
    if holdout:
        if generation is not None or evaluation:
            raise ValueError('holdout is a separate block')
        start, count = 1260000000, 3000
    else:
        if generation not in range(1, GENERATIONS + 1):
            raise ValueError('generation must be 1..3')
        start = (1250000000 if evaluation else 1240000000) + generation*100000
        count = 1000 if evaluation else GAMES
    if evaluation or holdout:
        return [(s, s%3, seat, 0) for s in range(start, start+count) for seat in range(3)]
    return [(start+i, i%3, (i//3)%3, 0) for i in range(count)]


def evaluate(directory, specs, models, initial, parent, reference, frozen, parent_hash, *, holdout=False):
    root = Path(directory)
    root.mkdir()
    states = {a: parameter_hash(m) for a, m in models.items()}
    initial_hash = parameter_hash(initial)
    policies = {'baseline': ExtractionPolicy(None, parent, reference, {'mode': 'baseline', 'lambda': .3}),
                'initial': ExtractionPolicy(initial, parent, reference, {'mode': 'anchor', 'lambda': .3})}
    for arm, model in models.items():
        policies[f'{arm}-anchor'] = ExtractionPolicy(model, parent, reference, {'mode': 'anchor', 'lambda': .3})
        if not holdout:
            policies[f'{arm}-raw'] = ExtractionPolicy(model, parent, reference, {'mode': 'raw', 'lambda': .3})
    outcomes, behavior = {}, {}
    for name, policy in policies.items():
        atomic_json(root / 'phase.json', {'phase': 'evaluation', 'policy': name, 'games': len(specs)})
        games, records, _ = collect(policy.eval(), parent, specs, root, name, device='cuda')
        outcomes[name] = games
        behavior[name] = behavior_summary(records)
        atomic_json(root / 'screening-outcomes.json', outcomes)
    atomic_json(root / 'phase.json', {'phase': 'full_replay_audit'})
    audit = audit_screening(root)
    atomic_json(root / 'phase.json', {'phase': 'seed_cluster_bootstrap'})
    baseline = {(g['seed'], g['seat']): g for g in outcomes['baseline']}
    a_reference = {(g['seed'], g['seat']): g for g in outcomes['A-anchor']}
    original = {(g['seed'], g['seat']): g for g in outcomes['initial']}
    cash = {name: cluster_cohort(games, baseline) for name, games in outcomes.items()}
    contrasts = {'B_minus_A': cluster_cohort(outcomes['B-anchor'], a_reference),
                 'B_minus_initial': cluster_cohort(outcomes['B-anchor'], original),
                 'A_minus_initial': cluster_cohort(outcomes['A-anchor'], original)}
    assert_frozen(frozen, parent, parent_hash)
    if parameter_hash(initial) != initial_hash or any(parameter_hash(m) != states[a] for a, m in models.items()):
        raise RuntimeError('evaluation changed weights/BN')
    result = {'cash': cash, 'contrasts': contrasts, 'behavior': behavior, 'audit': audit}
    atomic_json(root / 'result.json', result)
    atomic_json(root / 'phase.json', {'phase': 'complete'})
    return result


def run(directory):
    torch.set_num_threads(2)
    torch.manual_seed(20261015)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    if any(p.name != 'official-source' for p in root.iterdir()):
        raise RuntimeError('experiment output must be fresh apart from pinned source snapshot')
    official = root / 'official-source' / 'manifest.json'
    source = json.loads(official.read_text())
    if source['commit'] != '0cff2b52982be5b1163aa9a62fb01f03ce91e0d2':
        raise RuntimeError('wrong official commit')
    for name, info in source['files'].items():
        path = root / 'official-source' / (name.removeprefix('mortal/') if name.count('/') == 1 and name.endswith('.py') else name.replace('/', '__'))
        # model.py was fetched as an additional named snapshot.
        if not path.exists():
            path = root / 'official-source' / name.replace('/', '__')
        if digest(path) != info['sha256']:
            raise RuntimeError('official source snapshot changed')
    old = json.loads(Path('data/generated/mortal-online-v1/config.json').read_text())
    for p, sha in old['frozen_sha256'].items():
        if (p.startswith(('zimortal/engine/', 'zimortal/model/')) or p in (BASELINE, CHECKPOINT)) and digest(p) != sha:
            raise RuntimeError('fixed rules/model/start changed')
    frozen = source_hashes() | {p: digest(p) for p in (BASELINE, CHECKPOINT)}
    config = {'experiment': 'MC continuation only, matched single real root', 'generations': GENERATIONS,
              'games_per_arm_generation': GAMES, 'evaluation_seeds': 1000, 'holdout_seeds': 3000,
              'initial_checkpoint': CHECKPOINT, 'initial_checkpoint_sha256': digest(CHECKPOINT),
              'baseline': BASELINE, 'reference': old['behavior'], 'temperature': .5, 'lambda': .3,
              'A': 'full Boltzmann continuation; only matched first non-HU root is trained',
              'B': 'same single Boltzmann root; frozen current anchored greedy thereafter',
              'root_rule': 'first non-HU multi-choice; matching availability only; no return/action filtering',
              'scope_limit': 'early roots; not a test of all middle-game chi/peng states',
              'training': 'terminal cash/100 half-MSE only; online CQL0; no replay/offline/extra loss',
              'optimizer_protocol': old, 'matched_order': 'sort(seed,seat), same RNG permutation, batch256, partial last batch once',
              'final_candidate': 'Gen3 for both arms; no per-generation champion selection',
              'primary': 'B-anchor minus A-anchor on locked final holdout',
              'stable_positive': 'Gen2 and Gen3 AND final holdout B-A CI95 lower>0',
              'practical_equivalence': 'final B-A CI95 entirely within [-.5,+.5] yuan/game; scope limited to this pilot',
              'bootstrap': '10000 seed-cluster resamples; baseline/initial secondary context',
              'frozen_sha256': frozen, 'official_manifest_sha256': digest(official),
              'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(0)}
    atomic_json(root / 'config.json', config)
    atomic_json(root / 'seed-lock.json', {'generations': {str(g): {'train_A_B': seed_specs(g), 'evaluation': seed_specs(g, evaluation=True)}
                                                       for g in range(1, 4)}, 'holdout': seed_specs(holdout=True)})
    parent = load_model(BASELINE, 'cuda').requires_grad_(False)
    initial = load_model(CHECKPOINT, 'cuda').requires_grad_(False)
    models = {a: load_model(CHECKPOINT, 'cuda') for a in ('A', 'B')}
    if len({parameter_hash(m) for m in (*models.values(), initial)}) != 1:
        raise RuntimeError('initialization mismatch')
    optimizers = {a: make_optimizer(m) for a, m in models.items()}
    rngs = {a: np.random.default_rng(20261015) for a in models}
    updates = dict.fromkeys(models, 0)
    parent_hash, results = parameter_hash(parent), []
    for generation in range(1, 4):
        gen = root / f'gen-{generation:02d}'
        gen.mkdir()
        all_records, outcomes, manifests, producers, audits = {}, {}, {}, {}, {}
        for arm, model in models.items():
            armroot = gen / arm
            armroot.mkdir()
            data_root = armroot / 'data'
            data_root.mkdir()
            actor = ContinuationPolicy(model, parent, config['reference'], greedy_continuation=arm == 'B')
            producer = parameter_hash(model)
            producers[arm] = producer
            manifest = {'shards': [], 'arm': arm, 'generation': generation, 'producer_state_sha256': producer,
                        'all_rows_are_audit_evidence': True, 'trained_selection': 'matched-root-selection.json'}
            records, games = [], []
            specs = seed_specs(generation)
            for start in range(0, GAMES, 300):
                atomic_json(root / 'progress.json', {'phase': 'collection', 'generation': generation, 'arm': arm, 'completed': start, 'total': GAMES})
                shard = start//300
                batch_games, batch_records, data = collect(actor, parent, specs[start:start+300], armroot/f'evidence-{shard:02d}',
                                                          f'{arm}-generation-{generation}', training_rows=True, device='cuda')
                manifest['shards'].append(save_shard(data_root, shard, data))
                records.extend(batch_records)
                games.extend(batch_games)
                atomic_json(data_root / 'manifest.json', manifest)
            if parameter_hash(model) != producer:
                raise RuntimeError('collection changed frozen producer')
            atomic_json(armroot / 'training-outcomes.json', games)
            atomic_json(root / 'progress.json', {'phase': 'training_replay_audit', 'generation': generation, 'arm': arm})
            audits[arm] = audit_dataset(data_root, manifest)
            all_records[arm], outcomes[arm], manifests[arm] = records, games, manifest
            del actor, data, batch_records
        roots, selection = matched_roots(*(load_rows(gen/arm/'data', manifests[arm]) for arm in ('A', 'B')))
        atomic_json(gen / 'matched-root-selection.json', selection)
        cohort = {tuple(k) for k in selection['common_seed_seat']}
        if generation == 1:
            for a, b in zip(roots['A'], roots['B'], strict=True):
                if a['chosen'] != b['chosen'] or not np.array_equal(a['features'], b['features']) or not np.array_equal(a['actions'], b['actions']):
                    raise RuntimeError('Gen1 initial root/action mismatch')
        arms = {}
        for arm, model in models.items():
            records = all_records[arm]
            selected_records = [r for r in records if r['is_training_root'] and (r['seed'], r['seat']) in cohort]
            rows = roots[arm]
            if len(rows) != len(selected_records):
                raise RuntimeError('trained rows differ from traced roots')
            actual_by_key = {(r['seed'], r['seat']): r for r in selected_records}
            for row in rows:
                r = actual_by_key[int(row['seed']), int(row['seat'])]
                if row['history_step'] != r['step']-1 or row['chosen'] != r['chosen']:
                    raise RuntimeError('trained root not executed at traced step')
            if arm == 'B' and any(r['execution_mode'] == 'greedy_continuation' and (r['chosen'] != r['anchored_greedy'] or r['actual_behavior_probability'] != 1.) for r in records):
                raise RuntimeError('B continuation not greedy')
            before = regression(model, rows, device='cuda')
            q_before = q_probe(model, rows)
            parameters = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
            atomic_json(root / 'progress.json', {'phase': 'MC_update', 'generation': generation, 'arm': arm, 'training_rows': len(rows)})
            updates[arm], losses = update(model, optimizers[arm], rows, rngs[arm], updates[arm], device='cuda')
            drift = q_drift(q_before, q_probe(model, rows), rows)
            change = parameter_change(parameters, model)
            cp = gen / arm / 'checkpoint.pt'
            checkpoint(model, optimizers[arm], cp, updates[arm], rngs[arm], arm=arm, generation=generation,
                       producer_state_sha256=producers[arm], manifest_sha256=digest(gen/arm/'data'/'manifest.json'),
                       training_selection_sha256=digest(gen/'matched-root-selection.json'),
                       seed_lock_sha256=digest(root/'seed-lock.json'), source_sha256=frozen,
                       initial_checkpoint_sha256=digest(CHECKPOINT))
            atomic_json(gen/arm/'updates.json', losses)
            arms[arm] = {'checkpoint_sha256': digest(cp), 'updated_state_sha256': parameter_hash(model),
                         'producer_state_sha256': producers[arm], 'training_audit': audits[arm],
                         'training_rows': len(rows), 'all_decisions': len(records), 'updates': updates[arm],
                         'updates_this_generation': len(losses), 'all_behavior': behavior_summary(records),
                         'root_behavior': behavior_summary(selected_records),
                         'switch_triggers': sum(r['switch_trigger'] for r in records),
                         'all_root_draws': sum(r['is_training_root'] for r in records),
                         'vs_own_anchor_deviation': sum(r['changed_from_anchored_greedy'] for r in records)/max(1, len(records)),
                         'greedy_continuation_decisions': sum(r['execution_mode'] == 'greedy_continuation' for r in records),
                         'terminal_cash_yuan': distribution([g['cash'] for g in outcomes[arm]]),
                         'root_terminal_cash_yuan': distribution([r['terminal_cash']*100 for r in rows]),
                         'regression_before': before, 'regression_after': regression(model, rows, device='cuda'),
                         'parameter_change': change, 'q_drift': drift}
            atomic_json(gen/arm/'training-result.json', arms[arm])
            del parameters, q_before
        if updates['A'] != updates['B'] or rngs['A'].bit_generator.state != rngs['B'].bit_generator.state:
            raise RuntimeError('A/B optimizer budget or shuffle RNG differs')
        root_outcomes = {a: [g for g in outcomes[a] if (g['seed'], g['seat']) in cohort] for a in models}
        label_difference = cluster_cohort(root_outcomes['B'], {(g['seed'], g['seat']): g for g in root_outcomes['A']})
        atomic_json(root / 'progress.json', {'phase': 'evaluation', 'generation': generation})
        evaluation = evaluate(gen/'evaluation', seed_specs(generation, evaluation=True), models, initial, parent, config['reference'], frozen, parent_hash)
        summary = {'generation': generation, 'arms': arms, 'matched_root_count': len(cohort),
                   'training_cash_B_minus_A': label_difference, 'gen1_shared_root_actions_verified': generation == 1, **evaluation}
        results.append(summary)
        atomic_json(gen/'result.json', summary)
        atomic_json(root/'generations.json', results)
        print(json.dumps({'generation': generation, 'cash': evaluation['cash'], 'contrasts': evaluation['contrasts']}), flush=True)
        del roots, all_records
    atomic_json(root/'final-candidate-lock.json', {'generation': 3, 'A_sha256': digest(root/'gen-03/A/checkpoint.pt'),
                                                 'B_sha256': digest(root/'gen-03/B/checkpoint.pt'), 'seed_lock_sha256': digest(root/'seed-lock.json')})
    atomic_json(root/'progress.json', {'phase': 'final_holdout', 'generation': 3})
    final = evaluate(root/'final-holdout', seed_specs(holdout=True), models, initial, parent, config['reference'], frozen, parent_hash, holdout=True)
    lo, hi = final['contrasts']['B_minus_A']['seed_bootstrap_95']
    stable = lo > 0 and all(r['contrasts']['B_minus_A']['seed_bootstrap_95'][0] > 0 for r in results[-2:])
    atomic_json(root/'active-checkpoint.json', {'checkpoint': CHECKPOINT, 'sha256': digest(CHECKPOINT), 'baseline_replaced': False})
    atomic_json(root/'result.json', {'status': 'complete_stop', 'generations': results, 'final_holdout': final,
                                    'stable_B_improvement': bool(stable), 'practical_equivalence_half_yuan': lo > -.5 and hi < .5,
                                    'baseline_replaced': False, 'automatic_expansion': False})
    atomic_json(root/'progress.json', {'phase': 'complete_stop', 'generations': 3, 'stable_B_improvement': bool(stable)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='data/generated/mortal-continuation-v1')
    run(parser.parse_args().directory)
