"""Figures and per-device aggregation for the second-review revision run.

Every value plotted is read from results_revision2/ and is execution-derived. Same style
as figures_rerun.py; the calibration modes are the disjoint ones reported in the revision:
target_disjoint (first half of the held-out device's benign rows sets the threshold, the
second half measures the realised rate) and source_heldout (training-device benign rows
withheld from fitting). Each figure also writes a CSV of exactly the values it plots.

Usage:  python src/devicepulse/figures_revision2.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from devicepulse.figures import apply_style, BLUE, ORANGE, AQUA, SURFACE, INK, INK2, INK3, CUR, HIS

ROOT = Path(__file__).resolve().parents[2]
RES, FIG = ROOT / 'results_revision2', ROOT / 'figures_revision2'
LAB = {'gafgyt': 'BASHLITE to Mirai', 'mirai': 'Mirai to BASHLITE'}
MODES = [('target_disjoint', 'Threshold set on the target device'),
         ('source_heldout', 'Threshold set on training devices only')]


def hard3() -> pd.DataFrame:
    """Hard-protocol cells over the three seeds run in both directions."""
    d = pd.read_csv(RES / 'hard_cells.csv')
    return d[d.seed < 3]


def per_device_table() -> pd.DataFrame:
    """Twelve-seed per-device gain, BASHLITE to Mirai, budget 1e-4, target_disjoint."""
    d = pd.read_csv(RES / 'hard_cells.csv')
    d = d[(d.calibration == 'target_disjoint') & (d.train_family == 'gafgyt') &
          (d.false_alert_budget == 1e-4)]
    p = d.pivot_table(index=['held_out', 'seed'], columns='condition', values='campaign_recall')
    p['gain'] = p[HIS] - p[CUR]
    g = p.groupby('held_out')
    out = pd.DataFrame(dict(
        mean_current=g[CUR].mean(), mean_history=g[HIS].mean(),
        mean_gain=g.gain.mean(), median_gain=g.gain.median(),
        min_gain=g.gain.min(), max_gain=g.gain.max(), n_seeds=g.gain.size(),
    )).reset_index().sort_values('mean_gain')
    out.to_csv(RES / 'per_device_gain.csv', index=False)
    return out


def fig_calibration_gain(d, out: Path):
    apply_style(); fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=True)
    buds = sorted(d.false_alert_budget.unique()); x = np.arange(len(buds)); w = 0.34
    rows = []
    for ax, (cal, ttl) in zip(axes, MODES):
        g = d[d.calibration == cal]
        for j, (fam, col) in enumerate([('gafgyt', BLUE), ('mirai', ORANGE)]):
            t = g[g.train_family == fam].pivot_table(index='false_alert_budget',
                                                     columns='condition', values='campaign_recall')
            v = [(t.loc[b, HIS] - t.loc[b, CUR]) * 100 for b in buds]
            rows += [dict(calibration=cal, transfer=LAB[fam], false_alert_budget=b,
                          gain_points=vi) for b, vi in zip(buds, v)]
            ax.bar(x + (j - 0.5) * w, v, width=w * 0.9, color=col, edgecolor=SURFACE,
                   linewidth=1.0, zorder=3, label=LAB[fam])
            for xi, vi in zip(x + (j - 0.5) * w, v):
                ax.text(xi, vi + (0.6 if vi >= 0 else -2.2), f'{vi:.1f}', ha='center',
                        fontsize=7.2, color=INK)
        ax.axhline(0, color=INK2, lw=0.8, zorder=2)
        ax.set_xticks(x, [f'{b:g}' for b in buds])
        ax.set_xlabel('False-alert budget'); ax.set_title(ttl, fontsize=9.0)
        ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    axes[0].set_ylabel('Campaign-recall gain\nfrom causal history (points)')
    lo = min(r['gain_points'] for r in rows); hi = max(r['gain_points'] for r in rows)
    axes[0].set_ylim(min(-1.5, lo - 3), hi + 4)
    axes[1].legend(loc='upper left', ncols=1)
    fig.suptitle('The gain from causal history under disjoint calibration\n'
                 'mean over three seeds, threshold and realised rate on different rows',
                 fontsize=9.5, fontweight='bold')
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    pd.DataFrame(rows).to_csv(out.with_name(out.stem + '_table.csv'), index=False)


def fig_detection(c, out: Path, budget=1e-4):
    apply_style(); fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5), sharey=True)
    bands = [('within 10 rows', AQUA), ('11 to 1000 rows', BLUE),
             ('beyond 1000 rows', ORANGE), ('never detected', INK3)]
    rows = []
    for ax, fam in zip(axes, ['gafgyt', 'mirai']):
        g = c[(c.calibration == 'target_disjoint') & (c.false_alert_budget == budget) &
              (c.train_family == fam) & (c.seed < 3)]
        ys = []
        for cond in [CUR, HIS]:
            s = g[g.condition == cond]; n = len(s)
            det = s.detected.astype(bool); rr = s.rows_to_first_alert
            ys.append([100 * (det & (rr <= 10)).sum() / n,
                       100 * (det & (rr > 10) & (rr <= 1000)).sum() / n,
                       100 * (det & (rr > 1000)).sum() / n,
                       100 * (~det).sum() / n])
            rows.append(dict(transfer=LAB[fam], condition=cond, campaigns=n,
                             **{b[0]: v for b, v in zip(bands, ys[-1])}))
        y = np.arange(2)[::-1]
        for k, (lab, col) in enumerate(bands):
            left = [sum(ys[i][:k]) for i in range(2)]
            ax.barh(y, [ys[i][k] for i in range(2)], left=left, height=0.46, color=col,
                    edgecolor=SURFACE, linewidth=1.2, zorder=3, label=lab if fam == 'gafgyt' else None)
        for i, yy in enumerate(y):
            ax.text(103, yy, f'{ys[i][2]:.1f}% late\n{ys[i][3]:.1f}% missed',
                    va='center', fontsize=7.0, color=INK2, linespacing=1.4)
        ax.set_yticks(y, ['current only', 'history'], fontsize=8.4)
        ax.set_xlim(0, 136); ax.set_xticks([0, 50, 100])
        ax.set_xlabel('Share of attack campaigns (%)')
        ax.set_title(LAB[fam], fontsize=9.0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='outside lower center', ncols=4, fontsize=7.6)
    fig.suptitle('Where detection lands when undetected campaigns\nstay in the denominator',
                 fontsize=9.5, fontweight='bold')
    fig.savefig(out); plt.close(fig)
    pd.DataFrame(rows).to_csv(out.with_name(out.stem + '_table.csv'), index=False)


def fig_per_device(pdv, out: Path):
    apply_style(); fig, ax = plt.subplots(figsize=(6.8, 2.9))
    pdv = pdv.sort_values('mean_gain')
    y = np.arange(len(pdv))
    for yy, r in zip(y, pdv.itertuples()):
        ax.plot([r.min_gain * 100, r.max_gain * 100], [yy, yy], color=INK3, lw=1.6,
                solid_capstyle='round', zorder=2)
        ax.scatter(r.mean_gain * 100, yy, s=30, color=ORANGE, zorder=4, linewidths=0)
        ax.scatter(r.median_gain * 100, yy, s=30, facecolor=SURFACE, edgecolor=INK2,
                   linewidths=1.0, zorder=3)
    ax.axvline(0, color=INK2, lw=0.9, zorder=1)
    ax.set_yticks(y, [n.replace('_WHT_Security_Camera', '').replace('_Security_Camera', '')
                      .replace('_', ' ') for n in pdv.held_out], fontsize=7.8)
    ax.set_xlabel('Campaign-recall gain from causal history (points)')
    ax.set_ylim(-0.8, len(pdv) - 0.2)
    ax.xaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    ax.scatter([], [], s=30, color=ORANGE, linewidths=0, label=f'mean over {int(pdv.n_seeds.max())} seeds')
    ax.scatter([], [], s=30, facecolor=SURFACE, edgecolor=INK2, linewidths=1.0, label='median')
    ax.plot([], [], color=INK3, lw=1.6, label='range across seeds')
    ax.legend(loc='lower right', ncols=1)
    ax.set_title('The gain is concentrated in a minority of devices\n'
                 'BASHLITE to Mirai, budget 0.0001, threshold set on half the target benign rows',
                 fontsize=9.5)
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    pdv.to_csv(out.with_name(out.stem + '_table.csv'), index=False)


def main():
    FIG.mkdir(exist_ok=True)
    fig_calibration_gain(hard3(), FIG / 'figure_6_calibration.png')
    fig_detection(pd.read_csv(RES / 'hard_per_campaign.csv'), FIG / 'figure_8_detection.png')
    fig_per_device(per_device_table(), FIG / 'figure_9_per_device_rerun.png')
    print('figures written to', FIG)


if __name__ == '__main__':
    main()
