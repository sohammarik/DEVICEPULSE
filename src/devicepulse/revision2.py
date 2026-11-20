"""Second-review revision run (reviews received November 2026).

Answers the experimental requests of the second round, on one shared row sample:

  probe     identity probe per decay factor at equal dimension (23 features each),
            history-only, all features, per-device standardised features, under a
            random and a time-ordered split                            (R2 item 1)
  standard  random split, leave-one-device-out and a time-ordered split, 7 budgets,
            threshold set on one half of the held-out benign rows and the realised
            false-alert rate measured on the other half              (R2 items 2, 3, 5)
  hard      held-out device with held-out attack family, gradient boosting, four
            calibration modes on one fitted model per cell:
              target_same      threshold and realised rate on the same target rows
                               (the published rule, kept for comparison)
              target_disjoint  threshold on the first half of the target device's
                               benign rows in capture order, rate on the second half
              source_insample  threshold on training-device benign rows the model
                               was fitted on (the published source rule)
              source_heldout   threshold on 30 percent of training-device benign rows
                               withheld from fitting
            realised rate always measured on the target evaluation half, plus
            per-campaign first-alert positions.  BASHLITE to Mirai runs 12 seeds,
            Mirai to BASHLITE 3                                     (R1; R2 items 3, 4)
  norm      the hard protocol on per-device standardised features, 3 seeds (R2 item 2)
  ae        an unsupervised autoencoder detector trained on benign traffic only, in
            the N-BaIoT design, under the same calibration modes, 3 seeds (R2 item 6)

Sampling.  data.load_device seeds each device with Python's salted hash(), so the
published sample cannot be redrawn; this run fixes PYTHONHASHSEED=0 (checked below)
and every number it reports, baseline included, comes from that one sample.
Published results are not touched.  Everything lands in results_revision2/.

Usage:  PYTHONHASHSEED=0 python src/devicepulse/revision2.py sample|probe|standard|hard|norm|ae
"""
from __future__ import annotations
import json, os, platform, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from devicepulse.data import DECAYS, device_dirs, feature_columns, load_device
from devicepulse.experiment import CONDITIONS, TIGHT_BUDGETS, _model, campaign_recall
from devicepulse.rerun_review import tie_aware_threshold as _tie_aware_reference


def tie_aware_threshold(benign: np.ndarray, budget: float):
    """Same rule as rerun_review.tie_aware_threshold, by sorted search rather than a
    rescan per candidate, so continuous autoencoder scores stay tractable."""
    srt = np.sort(benign)
    u = np.unique(srt)
    cand = np.concatenate([u, [np.nextafter(u[-1], np.inf)]])
    rates = (len(srt) - np.searchsorted(srt, cand, side='left')) / len(srt)
    ok = np.flatnonzero(rates <= budget)
    if ok.size == 0:
        return float(cand[-1]), float(rates[-1]), False
    i = ok[0]
    return float(cand[i]), float(rates[i]), bool(cand[i] <= u[-1])

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results_revision2'; OUT.mkdir(exist_ok=True)
CACHE = ROOT / 'data_real' / 'revision2_sample.pkl'
STD_BUDGETS = (1e-4, 3e-4, 1e-3, 3e-3, 5e-3, 1e-2, 5e-2)
META = ('label', 'family', 'campaign', 'device', 'pos')


def log(m):
    line = f'{time.strftime("%H:%M:%S")}  {m}'
    print(line, flush=True)
    with open(OUT / 'revision2.log', 'a') as fh:
        fh.write(line + '\n')


def sample() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_pickle(CACHE)
    assert os.environ.get('PYTHONHASHSEED') == '0', 'run with PYTHONHASHSEED=0'
    frames, prov = [], []
    for d in device_dirs(ROOT / 'data_real/nbaiot'):
        f, p = load_device(d, 12000, 4000, seed=0)
        frames.append(f); prov += p
        log(f'  sampled {d.name}: {len(f)} rows')
    df = pd.concat(frames, ignore_index=True)
    # capture-order position within each file, so halves and time splits are defined
    df['pos'] = df.groupby('campaign').cumcount()
    df['nfile'] = df.groupby('campaign').campaign.transform('size')
    df.to_pickle(CACHE)
    pd.DataFrame(prov).to_csv(OUT / 'file_provenance.csv', index=False)
    return df


def first_half(rows: pd.DataFrame) -> np.ndarray:
    """Boolean over the given benign rows: True for the earlier half, in capture order, of
    each device's rows within this set (all of a device's rows under the hard protocol, its
    test rows under a split)."""
    rank = rows.groupby('device').pos.rank(method='first')
    n = rows.groupby('device').pos.transform('size')
    out = (rank <= n // 2).to_numpy()
    assert out.any() and (~out).any(), 'empty calibration or evaluation half'
    return out


def per_device_standardise(df, cols, stats_rows):
    """z-score each device's columns with mean/std from that device's stats_rows."""
    out = df[cols].to_numpy(dtype=float).copy()
    dev = df.device.to_numpy()
    for d in np.unique(dev):
        ref = stats_rows[stats_rows.device == d][cols].to_numpy(dtype=float)
        mu, sd = ref.mean(0), ref.std(0)
        sd[sd == 0] = 1.0
        out[dev == d] = (out[dev == d] - mu) / sd
    return out


# ------------------------------------------------------------------ probe
PROBE_SETS = {f'{d} only': (d,) for d in DECAYS}
PROBE_SETS['history only (L3, L1, L0.1, L0.01)'] = ('L3', 'L1', 'L0.1', 'L0.01')
PROBE_SETS['all five decays'] = DECAYS


def stage_probe(df):
    b = df[df.label == 0].reset_index(drop=True)
    rows = []
    for seed in (0, 1, 2):
        for split in ('random', 'time-ordered'):
            if split == 'random':
                tr_i, te_i = train_test_split(np.arange(len(b)), test_size=0.3,
                                              random_state=seed, stratify=b.device)
            else:
                late = (b.pos >= 0.7 * b.nfile).to_numpy()
                tr_i, te_i = np.flatnonzero(~late), np.flatnonzero(late)
            for norm in (False, True):
                for name, decays in PROBE_SETS.items():
                    cols = feature_columns(b, decays)
                    X = (per_device_standardise(b, cols, b.iloc[tr_i]) if norm
                         else b[cols].to_numpy())
                    y = b.device.to_numpy()
                    m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1,
                                                       random_state=seed)
                    m.fit(X[tr_i], y[tr_i])
                    pred = m.predict(X[te_i])
                    per_dev = pd.Series(pred == y[te_i]).groupby(y[te_i]).mean()
                    rows.append(dict(seed=seed, split=split, standardised=norm,
                                     feature_set=name, n_features=len(cols),
                                     accuracy=float(accuracy_score(y[te_i], pred)),
                                     chance=1 / b.device.nunique(),
                                     **{f'acc_{k}': float(v) for k, v in per_dev.items()}))
                    log(f'[probe] s{seed} {split} norm={norm} {name}: {rows[-1]["accuracy"]:.3f}')
                pd.DataFrame(rows).to_csv(OUT / 'identity_probe.csv', index=False)


# ------------------------------------------------------------------ calibration
def calibrations(s_tbenign_all, cal_half, s_src_in, s_src_out):
    """Return {mode: (pool used to set threshold)} plus the target evaluation scores."""
    pools = dict(target_same=s_tbenign_all,
                 target_disjoint=s_tbenign_all[cal_half])
    if s_src_in is not None:
        pools['source_insample'] = s_src_in
    if s_src_out is not None:
        pools['source_heldout'] = s_src_out
    return pools, s_tbenign_all[~cal_half]


# ------------------------------------------------------------------ standard
def stage_standard(df):
    rows = []
    for seed in (0, 1, 2):
        splits = []
        for dev in sorted(df.device.unique()):
            splits.append(('leave-one-device-out', dev, df.device != dev, df.device == dev))
        tr_i, _ = train_test_split(np.arange(len(df)), test_size=0.3, random_state=seed,
                                   stratify=df.label)
        rnd = np.zeros(len(df), bool); rnd[tr_i] = True
        splits.append(('random split', 'none', pd.Series(rnd, df.index),
                       pd.Series(~rnd, df.index)))
        early = df.pos < 0.7 * df.nfile
        splits.append(('time-ordered split', 'none', early, ~early))
        for proto, held, trm, tem in splits:
            tr, te = df[trm.to_numpy()], df[tem.to_numpy()]
            for cond in CONDITIONS:
                cols = feature_columns(tr, CONDITIONS[cond])
                m = _model('trees', seed).fit(tr[cols].to_numpy(), tr.label.to_numpy())
                s = m.predict_proba(te[cols].to_numpy())[:, 1]
                y = te.label.to_numpy()
                tb = te[y == 0]
                cal = first_half(tb)
                pools, s_eval = calibrations(s[y == 0], cal, None, None)
                for b in STD_BUDGETS:
                    for mode, pool in pools.items():
                        thr, _, att = tie_aware_threshold(pool, b)
                        rec, hit, tot = campaign_recall(s, y, te.campaign.to_numpy(), thr)
                        rows.append(dict(protocol=proto, held_out=held, condition=cond,
                                         seed=seed, false_alert_budget=b, calibration=mode,
                                         nominal_attainable=att,
                                         realised_rate_eval=float((s_eval >= thr).mean()),
                                         realised_rate_all=float((s[y == 0] >= thr).mean()),
                                         campaign_recall=rec, campaigns_detected=hit,
                                         campaigns=tot,
                                         row_recall=float((s[y == 1] >= thr).mean())))
            log(f'[std] s{seed} {proto} {held} done')
        pd.DataFrame(rows).to_csv(OUT / 'standard_protocol.csv', index=False)


# ------------------------------------------------------------------ hard
def hard_cell(df, dev, seed, trf, tefs, cond, model, norm):
    """One fitted model for one held-out device; all four calibration modes, scored on
    each test family in tefs (the autoencoder never sees attacks, so one fit serves both)."""
    rng = np.random.default_rng(seed)
    train_fams = ['benign'] if model == 'ae' else ['benign', trf]
    tr_all = df[(df.device != dev) & df.family.isin(train_fams)]
    te = df[(df.device == dev) & (df.family == 'benign')]
    # withhold 30 percent of each training device's benign rows for source calibration
    tb = tr_all[tr_all.label == 0]
    hold = np.zeros(len(tb), bool)
    for d in tb.device.unique():
        idx = np.flatnonzero((tb.device == d).to_numpy())
        hold[rng.choice(idx, size=int(round(0.3 * len(idx))), replace=False)] = True
    src_cal = tb[hold]
    tr = tr_all.drop(index=src_cal.index)
    cols = feature_columns(tr, CONDITIONS[cond])
    tbe = te
    cal = first_half(tbe)
    if norm:
        stats = pd.concat([tr[tr.label == 0], tbe[cal]])
        X = lambda rows: per_device_standardise(rows, cols, stats)
    else:
        X = lambda rows: rows[cols].to_numpy()
    Xtr, Xsi, Xso = X(tr), X(tr[tr.label == 0]), X(src_cal)
    if model == 'ae':
        sc = StandardScaler().fit(Xtr)
        d = len(cols)
        h = tuple(max(2, int(round(f * d))) for f in (.75, .5, .33, .25, .33, .5, .75))
        m = MLPRegressor(hidden_layer_sizes=h, batch_size=256, max_iter=60,
                         early_stopping=True, n_iter_no_change=5, random_state=seed)
        Ztr = np.clip(sc.transform(Xtr), -50, 50)
        m.fit(Ztr, Ztr)

        def score(X):
            Z = np.clip(sc.transform(X), -50, 50)
            return ((m.predict(Z) - Z) ** 2).mean(1)
    else:
        m = _model('trees', seed).fit(Xtr, tr.label.to_numpy())

        def score(X):
            return m.predict_proba(X)[:, 1]
    s_ben = score(X(tbe))
    pools, s_eval = calibrations(s_ben, cal, score(Xsi), score(Xso))
    cells, percamp = [], []
    for tef in tefs:
        ta = df[(df.device == dev) & (df.family == tef)]
        camps, s_att = ta.campaign.to_numpy(), score(X(ta))
        base = dict(held_out=dev, condition=cond, seed=seed, model=model,
                    standardised=norm, test_family=tef,
                    train_family='benign only' if model == 'ae' else trf)
        _score_cells(base, pools, s_ben, s_eval, camps, s_att, cells, percamp)
    return cells, percamp


def _score_cells(base, pools, s_ben, s_eval, camps, s_att, cells, percamp):
    for b in TIGHT_BUDGETS:
        for mode, pool in pools.items():
            thr, _, attain = tie_aware_threshold(pool, b)
            det, pos = [], []
            for c in pd.unique(camps):
                f = np.flatnonzero(s_att[camps == c] >= thr)
                det.append(bool(f.size)); pos.append(int(f[0]) + 1 if f.size else -1)
                percamp.append(dict(base, false_alert_budget=b, calibration=mode,
                                    campaign=c, detected=bool(f.size),
                                    rows_to_first_alert=pos[-1]))
            det = np.array(det)
            cells.append(dict(base, false_alert_budget=b, calibration=mode,
                              nominal_attainable=attain,
                              realised_rate_eval=float((s_eval >= thr).mean()),
                              realised_rate_all_target=float((s_ben >= thr).mean()),
                              n_eval_benign=int(len(s_eval)), n_cal_pool=int(len(pool)),
                              campaign_recall=float(det.mean()), campaigns=int(det.size),
                              campaigns_detected=int(det.sum()),
                              row_recall=float((s_att >= thr).mean())))


def stage_hard(df, model='trees', norm=False, tag='hard'):
    cells, percamp = [], []
    plan = []
    if model == 'ae':
        plan = [(s, 'benign', None) for s in (0, 1, 2)]
    else:
        seeds_bm = range(12) if not norm else range(3)
        plan = ([(s, 'gafgyt', 'mirai') for s in seeds_bm] +
                [(s, 'mirai', 'gafgyt') for s in (0, 1, 2)])
    t0 = time.time()
    for seed, trf, tef in plan:
        for dev in sorted(df.device.unique()):
            tefs = [f for f in (('mirai', 'gafgyt') if model == 'ae' else (tef,))
                    if not df[(df.device == dev) & (df.family == f)].empty]
            if not tefs:
                continue
            for cond in CONDITIONS:
                c, pc = hard_cell(df, dev, seed, trf, tefs, cond, model, norm)
                cells += c; percamp += pc
            log(f'[{tag}] s{seed} {trf}->{tef} {dev} ({time.time()-t0:.0f}s)')
        pd.DataFrame(cells).to_csv(OUT / f'{tag}_cells.csv', index=False)
        pd.DataFrame(percamp).to_csv(OUT / f'{tag}_per_campaign.csv', index=False)


def main():
    stage = sys.argv[1]
    df = sample()
    log(f'stage {stage}: sample {len(df)} rows, {df.device.nunique()} devices')
    if stage == 'probe':
        stage_probe(df)
    elif stage == 'standard':
        stage_standard(df)
    elif stage == 'hard':
        stage_hard(df)
    elif stage == 'norm':
        stage_hard(df, norm=True, tag='norm')
    elif stage == 'ae':
        stage_hard(df, model='ae', tag='ae')
    man = OUT / 'manifest.json'
    m = json.loads(man.read_text()) if man.exists() else {}
    m[stage] = dict(completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    pythonhashseed=os.environ.get('PYTHONHASHSEED'),
                    python=platform.python_version(), sklearn=sklearn.__version__,
                    numpy=np.__version__, pandas=pd.__version__,
                    platform=platform.platform(), cpu_count=os.cpu_count())
    man.write_text(json.dumps(m, indent=1))
    log(f'stage {stage} DONE')


if __name__ == '__main__':
    main()
