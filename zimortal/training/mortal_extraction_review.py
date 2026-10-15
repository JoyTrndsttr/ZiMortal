"""Read-only audit and report for dev-locked fixed-Q extraction; no fitting."""

import json
from pathlib import Path

from zimortal.training.active import atomic_json
from zimortal.training.online_rl import digest


def run():
    root = Path('data/generated/mortal-extraction-v1')
    result = json.loads((root / 'result.json').read_text())
    lock = json.loads((root / 'selection-lock.json').read_text())
    config = json.loads((root / 'config.json').read_text())
    if result['training_updates'] != 0 or result['online_started'] or result['selected'] != lock['selected']:
        raise RuntimeError('fixed-Q/selection protocol mismatch')
    if digest(config['checkpoint']) != lock['checkpoint_sha256'] or digest(root / 'seed-lock.json') != lock['seed_lock_sha256']:
        raise RuntimeError('checkpoint or locked seeds changed')
    if result['development']['network_state_sha256'] != result['test']['network_state_sha256']:
        raise RuntimeError('network parameters/buffers changed between splits')
    for path, sha in config['frozen_sha256'].items():
        if digest(path) != sha:
            raise RuntimeError('frozen source/checkpoint changed')
    main = {'A-raw': 'A-raw', 'B-support': 'B-support', **result['selected']}
    cash = result['test']['comparisons']
    raw_cash = cash['A-raw']['0']['mean_cash_delta']
    table = {}
    family_rows = []
    for label, name in main.items():
        v = cash[name]['0']
        delta = result['comparisons'][name]['vs_raw']['0']
        table[label] = {
            'locked_policy': name, 'vs_baseline': v, 'vs_raw': delta,
            'raw_loss_recovered_fraction_point': (v['mean_cash_delta'] - raw_cash) / -raw_cash if raw_cash < 0 else None,
        }
        for family, d in result['test']['diagnostics'][name].items():
            if family == 'all':
                continue
            family_rows.append({'policy': label, 'locked_policy': name, 'family': family, **d})
    atomic_json(root / 'review.json', {
        'cash': table, 'family_statistics': family_rows,
        'family_ablation': result['family_ablation'],
        'anchor_vs_support': result['comparisons'][main['D-anchor']]['vs_support'],
        'dev_state_sha256': result['development']['network_state_sha256'],
        'test_state_sha256': result['test']['network_state_sha256'],
        'training_updates': 0, 'online_started': False,
        'not_a_formal_promotion': True,
    })
    def cash_line(label, v):
        lo, hi = v['paired_seed_bootstrap_95']
        return f'|{label}|{v["mean_cash_delta"]:+.2f}|[{lo:.2f},{hi:.2f}]|'
    lines = ['|策略|相对baseline元/局|95%CI|', '|---|---:|---|']
    for label, row in table.items():
        lines.append(cash_line(f'{label}（{row["locked_policy"]}）', row['vs_baseline']))
    lines += ['', '|消融（现金差）|元/局|95%CI|', '|---|---:|---|']
    for name, v in result['family_ablation'].items():
        lines.append(cash_line(name, v['0']))
    lines += ['', '|策略|动作族|改动作率|推翻中低支持|父实际选择率|Q margin均值（元）|', '|---|---|---:|---:|---:|---:|']
    for r in family_rows:
        low = r['low_support_among_overrides']
        low_s = '—' if low is None else f'{low*100:.2f}%'
        margin = r['q_margin'].get('mean')
        margin_s = '—' if margin is None else f'{margin*100:.3f}'
        lines.append(f'|{r["policy"]}|{r["family"]}|{r["changed_action_rate"]*100:.2f}%|{low_s}|{r["parent_selection_rate"]*100:.2f}%|{margin_s}|')
    (root / 'review-tables.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(table, indent=2))
    print('ABLATION', json.dumps(result['family_ablation'], indent=2))
    print('ANCHOR_VS_SUPPORT', json.dumps(result['comparisons'][main['D-anchor']]['vs_support']))


if __name__ == '__main__':
    run()
