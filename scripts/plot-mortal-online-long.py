"""Plot the full preregistered online curve, without selecting a checkpoint."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['svg.hashsalt'] = 'mortal-online-long-v1'
import matplotlib.pyplot as plt
import numpy as np


def plot(root, output):
    generations = json.loads((root / 'generations.json').read_text())
    x = np.array([r['generation'] for r in generations])
    fig, axes = plt.subplots(4, 2, figsize=(14, 16), constrained_layout=True)
    axes = axes.ravel()

    def cash_curve(ax, rows, label, color):
        y = [r['paired_delta'] for r in rows]
        lo, hi = np.array([r['seed_bootstrap_95'] for r in rows]).T
        ax.plot(x, y, 'o-', label=label, color=color)
        ax.fill_between(x, lo, hi, alpha=.15, color=color)
        ax.set_ylabel('Paired cash / game (yuan), seed-cluster 95% CI')

    cash_curve(axes[0], [r['cash']['anchor'] for r in generations], 'Current anchor - baseline', '#117733')
    cash_curve(axes[0], [r['cash']['initial'] for r in generations], 'Initial anchor - baseline', '#0077bb')
    cash_curve(axes[1], [r['contrasts']['anchor_minus_initial'] for r in generations], 'Current anchor - initial anchor', '#0077bb')
    cash_curve(axes[2], [r['cash']['raw'] for r in generations], 'Raw Q - baseline', '#cc3311')
    cash_curve(axes[2], [r['contrasts']['raw_minus_anchor'] for r in generations], 'Raw Q - current anchor', '#ee7733')
    for ax in axes[:2]:
        ax.axhline(0, color='gray', lw=1)
    t = [r['training'] for r in generations]
    axes[3].plot(x, [v['behavior']['all']['changed_action_rate']*100 for v in t], 'o-', label='Boltzmann train deviations')
    for key in ('anchor', 'raw'):
        axes[3].plot(x, [r['behavior'][key]['all']['changed_action_rate']*100 for r in generations], 'o-', label=f'{key} evaluation deviations')
    axes[3].plot(x, [r['behavior']['raw']['all']['low_support_among_overrides']*100 for r in generations], 'o--', label='Raw overrides with p_ref <= .1')
    axes[3].set_ylabel('Percent')
    axes[4].plot(x, [v['diagnostics']['all']['entropy']['mean'] for v in t], 'o-', label='Train Boltzmann entropy (nats)')
    axes[4].plot(x, [v['behavior']['all']['actual_chosen_probability']['mean'] for v in t], 'o-', label='Actual chosen probability')
    axes[4].plot(x, [v['behavior']['all']['chosen_p_reference']['mean'] for v in t], 'o-', label='Chosen p_reference')
    for key in ('before', 'after'):
        axes[5].plot(x, [v[f'regression_{key}']['rmse'] for v in t], 'o-', label=f'Current training rows RMSE {key}')
    axes[5].set_ylabel('Cash / 100 (training fit only)')
    axes[6].plot(x, [v['parameter_change']['relative_delta_l2']*100 for v in t], 'o-', label='Relative parameter L2 update (%)')
    axes[6].set_ylabel('Relative parameter update (%)')
    axes[7].plot(x, [v['q_drift']['legal_q_rmse'] for v in t], 'o-', color='#ee7733', label='Legal Q output drift RMSE')
    axes[7].set_ylabel('Cash / 100 (same current-generation training inputs)')
    for ax in axes:
        ax.set_xlabel('Generation (fixed final candidate: Gen10)')
        ax.set_xticks(x)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    final = root / 'final-holdout' / 'result.json'
    if final.exists():
        r = json.loads(final.read_text())['cash']['anchor']
        fig.suptitle(f"Long-timescale online: Gen10 holdout delta {r['paired_delta']:+.3f} yuan, 95% CI [{r['seed_bootstrap_95'][0]:+.3f}, {r['seed_bootstrap_95'][1]:+.3f}]\nFixed schedule reaches LR=3e-6 at update 320; no intermediate checkpoint selection", fontsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == '.svg':
        fig.savefig(output, metadata={'Date': None})
        output.write_text('\n'.join(line.rstrip() for line in output.read_text().splitlines()) + '\n')
    else:
        fig.savefig(output)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=Path('data/generated/mortal-online-long-v1'))
    parser.add_argument('--output', type=Path, default=Path('docs/training/figures/mortal-online-long-v1.svg'))
    args = parser.parse_args()
    plot(args.directory, args.output)
