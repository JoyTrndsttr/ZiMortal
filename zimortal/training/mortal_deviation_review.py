"""Read-only reporting of predeclared deviations; never selects an actor."""

import csv
import gzip
import json
from pathlib import Path

from zimortal.training.active import atomic_json
from zimortal.training.online_rl import digest


def run():
    root = Path('data/generated/mortal-deviation-v1')
    result = json.loads((root / 'result.json').read_text())
    config = json.loads((root / 'config.json').read_text())
    if result['training_updates'] != 0 or result['online_started']:
        raise RuntimeError('not a frozen zero-update experiment')
    for path, sha in config['frozen_sha256'].items():
        if digest(path) != sha:
            raise RuntimeError('frozen source/checkpoint changed')
    comparisons = {name: result['comparisons'][name]['0'] for name in config['budgets']}
    # Positive signals are listed, never used to choose a budget or replace baseline.
    signals = [n for n, c in comparisons.items() if n != 'baseline' and c['paired_seed_bootstrap_95'][0] > 0]
    simultaneous = [n for n in signals if comparisons[n]['paired_seed_bootstrap_interval'][0] > 0]
    audit_counts = {}
    for name in ('anchor-all', 'anchor-first', 'anchor-max2'):
        games = json.loads((root / f'game-deviations-{name}.json').read_text())
        indexed = {(g['seed'], g['seat']): g for g in games}
        counts = {k: 0 for k in indexed}
        fields = ['seed', 'seat', 'step', 'family', 'chosen_kind', 'parent_kind',
                  'p_behavior', 'q_selected_minus_parent', 'anchor_score_margin',
                  'terminal_cash', 'baseline_cash', 'paired_game_delta', 'game_deviations',
                  'deviation_count_after']
        with gzip.open(root / f'decisions-{name}.jsonl.gz', 'rt') as source, (root / f'deviations-{name}.csv').open('w') as target:
            writer = csv.DictWriter(target, fieldnames=fields)
            writer.writeheader()
            for line in source:
                r = json.loads(line)
                if not r['changed']:
                    continue
                key = (r['seed'], r['seat'])
                g = indexed[key]
                counts[key] += 1
                if r['terminal_cash'] != g['cash']:
                    raise RuntimeError('deviation cash does not match game')
                writer.writerow({
                    'seed': r['seed'], 'seat': r['seat'], 'step': r['step'], 'family': r['family'],
                    'chosen_kind': r['actions'][r['chosen']]['kind'],
                    'parent_kind': r['actions'][r['parent']]['kind'],
                    'p_behavior': r['behavior_probability'],
                    'q_selected_minus_parent': r['q_selected_minus_parent'],
                    'anchor_score_margin': r['unlimited_anchor_margin'],
                    'terminal_cash': r['terminal_cash'], 'baseline_cash': g['baseline_cash'],
                    'paired_game_delta': g['cash'] - g['baseline_cash'],
                    'game_deviations': g['deviations'], 'deviation_count_after': r['deviation_count_after'],
                })
        if any(counts[k] != g['deviations'] for k, g in indexed.items()):
            raise RuntimeError('exported deviation counts mismatch')
        audit_counts[name] = sum(counts.values())
    atomic_json(root / 'review.json', {
        'nominal_positive_signal_policies': signals,
        'simultaneous_positive_signal_policies': simultaneous,
        'no_policy_selected_on_test': True, 'exported_actual_deviations': audit_counts,
        'training_updates': 0, 'online_started': False,
    })
    def ci(x):
        return '—' if x is None else f'[{x[0]:.2f},{x[1]:.2f}]'
    lines = ['|固定策略|相对baseline元/局|95%CI|98.33%诊断区间|', '|---|---:|---|---|']
    for name, c in comparisons.items():
        lines.append(f'|{name}|{c["mean_cash_delta"]:+.3f}|{ci(c["paired_seed_bootstrap_95"])}|{ci(c["paired_seed_bootstrap_interval"])}|')
    lines += ['', '|策略|偏离次数/决策占比|有偏离局|一次局|多次局|', '|---|---:|---:|---:|---:|']
    for name, stats in result['statistics'].items():
        cc = stats['count_cohorts']
        lines.append(f'|{name}|{stats["deviations"]} / {stats["deviation_rate"]*100:.2f}%|{cc["once"]["games"]+cc["multiple"]["games"]}|{cc["once"]["games"]}|{cc["multiple"]["games"]}|')
    lines += ['', '|策略|偏离次数组|局数|平均现金|同局baseline现金|配对差|95%CI（条件描述）|', '|---|---|---:|---:|---:|---:|---|']
    for name, stats in result['statistics'].items():
        if name == 'baseline':
            continue
        for label, c in stats['count_cohorts'].items():
            if c['games']:
                lines.append(f'|{name}|{label}|{c["games"]}|{c["mean_cash"]:.2f}|{c["mean_baseline_cash_same_games"]:.2f}|{c["paired_delta"]:+.2f}|{ci(c["seed_bootstrap_95"])}|')
    stats = result['statistics']['anchor-all']
    lines += ['', '|anchor-all动作族|偏离数|不同局|平均p|平均Q差（原单位）|平均anchor margin|局去重配对差|95%CI（条件描述）|', '|---|---:|---:|---:|---:|---:|---:|---|']
    for family, g in stats['family'].items():
        c = g['unique_game_cash']
        if not g['decisions']:
            lines.append(f'|{family}|0|0|—|—|—|—|—|')
        else:
            lines.append(f'|{family}|{g["decisions"]}|{c["games"]}|{g["p_behavior"]["mean"]:.3f}|{g["q_delta"]["mean"]:.4f}|{g["anchor_margin"]["mean"]:.4f}|{c["paired_delta"]:+.2f}|{ci(c["seed_bootstrap_95"])}|')
    for name, buckets in stats['buckets'].items():
        lines += ['', f'anchor-all预声明{name}桶（条件描述，不是动作EV）：', '', '|区间|偏离数|不同局|局去重平均现金|同局配对差|95%CI|', '|---|---:|---:|---:|---:|---|']
        for label, g in buckets.items():
            c = g['unique_game_cash']
            if not g['decisions']:
                lines.append(f'|{label}|0|0|—|—|—|')
            else:
                lines.append(f'|{label}|{g["decisions"]}|{c["games"]}|{c["mean_cash"]:.2f}|{c["paired_delta"]:+.2f}|{ci(c["seed_bootstrap_95"])}|')
    (root / 'review-tables.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'cash': comparisons, 'budget_contrasts': result['budget_contrasts'],
                      'positive': signals, 'adjusted_positive': simultaneous}, indent=2))


if __name__ == '__main__':
    run()
