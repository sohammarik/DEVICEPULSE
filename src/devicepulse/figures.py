"""Figures for DEVICEPULSE from executed artifacts. No execution time reported."""
from __future__ import annotations
from pathlib import Path
import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np, pandas as pd
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
SURFACE, INK, INK2, INK3, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8982', '#e4e3dd'
CUR, HIS = 'current window only', 'current window plus causal history'
def apply_style():
    mpl.rcParams.update({'figure.facecolor':SURFACE,'axes.facecolor':SURFACE,
      'savefig.facecolor':SURFACE,'savefig.dpi':400,'figure.dpi':140,
      'font.family':'DejaVu Sans','font.size':8.5,'axes.edgecolor':INK3,
      'axes.linewidth':0.6,'axes.labelcolor':INK,'axes.titlecolor':INK,
      'axes.titlesize':9.5,'axes.titleweight':'bold','axes.titlelocation':'left',
      'axes.titlepad':8,'axes.labelsize':8.5,'axes.spines.top':False,
      'axes.spines.right':False,'xtick.color':INK2,'ytick.color':INK2,
      'xtick.labelsize':8,'ytick.labelsize':8,'grid.color':GRID,'grid.linewidth':0.6,
      'legend.frameon':False,'legend.fontsize':8,'legend.labelcolor':INK2,
      'lines.linewidth':2.0,'figure.constrained_layout.use':True})

def fig_saturation(r, out: Path):
    """The standard protocol cannot separate anything: every cell is 1.000."""
    t = r.groupby(['protocol','condition','false_alert_budget']).campaign_recall.mean().reset_index()
    apply_style(); fig, ax = plt.subplots(figsize=(6.6,2.7))
    prot = sorted(t.protocol.unique()); conds=[CUR,HIS]
    x = np.arange(len(t.false_alert_budget.unique())); buds=sorted(t.false_alert_budget.unique())
    w=0.2
    for j,(p,c) in enumerate([(p,c) for p in prot for c in conds]):
        v=[t[(t.protocol==p)&(t.condition==c)&(t.false_alert_budget==b)].campaign_recall.iloc[0] for b in buds]
        ax.bar(x+(j-1.5)*w, v, width=w*0.9, color=[BLUE,AQUA,ORANGE,INK3][j],
               edgecolor=SURFACE, linewidth=1.0, zorder=3,
               label=f"{'LODO' if 'leave' in p else 'random split'} · {'history' if c==HIS else 'current only'}")
    ax.axhline(1.0, color=INK2, lw=0.9, ls=(0,(4,3)), zorder=1)
    ax.set_xticks(x,[f'{b:g}' for b in buds]); ax.set_ylim(0,1.12)
    ax.set_xlabel('False-alert budget'); ax.set_ylabel('Campaign recall')
    ax.set_title('The standard protocol is saturated: every condition detects every campaign\n'
                 'nine real IoT devices, both protocols, all budgets', fontsize=9.5)
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    ax.legend(loc='lower left', bbox_to_anchor=(0,-0.46), ncols=2)
    fig.savefig(out, bbox_inches='tight'); plt.close(fig); return t

def fig_identity(pr, out: Path):
    """Device identity recovered from benign traffic alone."""
    t = pr.groupby('condition')[['device_identification_accuracy','chance']].mean().reindex([CUR,HIS])
    apply_style(); fig, ax = plt.subplots(figsize=(6.6,2.1))
    y=np.arange(len(t))[::-1]
    for yy,(c,row) in zip(y,t.iterrows()):
        ax.barh(yy,row.device_identification_accuracy,height=0.42,
                color=BLUE if c==CUR else ORANGE, edgecolor=SURFACE, linewidth=1.1, zorder=3)
        ax.text(row.device_identification_accuracy+0.012, yy, f'{row.device_identification_accuracy:.3f}',
                va='center', fontsize=8, color=INK)
    ax.axvline(t.chance.iloc[0], color=INK2, lw=0.9, ls=(0,(4,3)), zorder=1)
    ax.text(t.chance.iloc[0]+0.012, -0.72, 'chance 0.111 (nine devices)', fontsize=7.4, color=INK2)
    ax.set_yticks(y,['current window only','plus causal history'],fontsize=8.4)
    ax.set_xlim(0,1.1); ax.set_ylim(-1.0,len(t)-0.4)
    ax.set_xlabel('Accuracy identifying which device produced a benign sample')
    ax.set_title('Causal history carries a device fingerprint', fontsize=9.5)
    ax.xaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    fig.savefig(out, bbox_inches='tight'); plt.close(fig); return t

def fig_hard_timeliness(h, out: Path):
    """Detection latency: the median is unchanged; the tail is what moves."""
    ok = h[h.status == 'OK']
    g = ok.groupby(['train_family', 'condition', 'false_alert_budget']).median_rows_to_detection
    med = g.median(); slow = g.apply(lambda v: (v > 10).mean())
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.9))
    buds = sorted(ok.false_alert_budget.unique())
    for tf, mk, ls in (('mirai', 'o', '-'), ('gafgyt', 's', '--')):
        for cond, col in ((CUR, BLUE), (HIS, ORANGE)):
            lab = f"{'current' if cond == CUR else 'history'} · {tf}\u2192{'gafgyt' if tf=='mirai' else 'mirai'}"
            ax1.plot(buds, [med.loc[(tf, cond, b)] for b in buds], color=col, marker=mk,
                     ls=ls, lw=1.8, markeredgecolor=SURFACE, markeredgewidth=0.8,
                     zorder=3, label=lab)
            ax2.plot(buds, [slow.loc[(tf, cond, b)] for b in buds], color=col, marker=mk,
                     ls=ls, lw=1.8, markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=3)
    for ax, ylab, title in ((ax1, 'Median rows before\nfirst detection',
                             'Typical latency is unchanged'),
                            (ax2, 'Share of runs needing\nmore than ten rows',
                             'The tail is what moves')):
        ax.set_xscale('log'); ax.set_xticks(buds, [f'{b:g}' for b in buds], fontsize=7)
        ax.set_xlabel('False-alert budget', fontsize=8)
        ax.set_ylabel(ylab, fontsize=8); ax.set_title(title, fontsize=9)
        ax.grid(True, zorder=0); ax.set_axisbelow(True)
    ax1.set_ylim(0, 3.2)
    ax1.legend(loc='upper left', bbox_to_anchor=(0, -0.34), ncols=2, fontsize=6.8)
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    return pd.DataFrame(dict(median=med, share_slow=slow))


def fig_per_device(h, out: Path, budget=1e-4):
    """Per-device gain, because the mean hides a loss and a large win."""
    ok=h[(h.status=='OK')&(h.train_family=='gafgyt')&(h.false_alert_budget==budget)]
    q=ok.pivot_table(index='held_out',columns='condition',values='campaign_recall')
    q['gain']=(q[HIS]-q[CUR])*100
    q=q.sort_values('gain')
    apply_style(); fig, ax = plt.subplots(figsize=(6.6,3.0))
    y=np.arange(len(q))
    cols=[ORANGE if v<0 else (INK3 if v==0 else BLUE) for v in q.gain]
    ax.barh(y,q.gain,height=0.56,color=cols,edgecolor=SURFACE,linewidth=1.0,zorder=3)
    for yy,v in zip(y,q.gain):
        ax.text(v+(1.5 if v>=0 else -1.5), yy, f'{v:+.1f}', va='center',
                ha='left' if v>=0 else 'right', fontsize=7.6, color=INK)
    ax.axvline(0,color=INK2,lw=0.9,zorder=2)
    ax.axvline(q.gain.mean(),color=INK2,lw=0.9,ls=(0,(4,3)),zorder=1)
    ax.text(q.gain.mean(), len(q)-0.3, f' mean {q.gain.mean():+.1f}', fontsize=7.4, color=INK2)
    ax.set_yticks(y,[i.replace('_',' ')[:30] for i in q.index],fontsize=7.4)
    ax.set_xlabel('Campaign-recall gain from causal history (percentage points)')
    ax.set_title('The mean gain is one large loss, one large win and five ties\n'
                 'held-out device, unseen attack family, budget 1e-4', fontsize=9.5)
    ax.xaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    fig.savefig(out, bbox_inches='tight'); plt.close(fig); return q
