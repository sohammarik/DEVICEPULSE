"""Rerun answering the external review of 13 October 2026.

Four changes against src/devicepulse/experiment.py, nothing else:

  1. Tie-aware threshold. The original took a quantile of benign scores. Scores are
     discrete, so `benign >= quantile` could realise far more than the nominal budget.
     Here the threshold is the lowest score at which the realised rate still satisfies
     the budget, and the run records whether the nominal point was attainable at all.
  2. Two calibration modes, reported separately and never mixed.
       oracle  threshold from the held-out device's own benign scores (the original)
       source  threshold from the training devices' benign scores (no target access)
  3. Realised false-alert rate kept for every cell, as before, now beside both modes.
  4. Per-campaign detection records written out, so detection can be reported against
     all campaigns rather than conditioned on the ones that were detected.

Original results are not touched. Everything lands in results_rerun/.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from devicepulse.data import device_dirs, load_device
from devicepulse.experiment import CONDITIONS, _model, TIGHT_BUDGETS
from devicepulse.data import feature_columns

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results_rerun'; OUT.mkdir(exist_ok=True)
CACHE = OUT / 'sample_cache.pkl'

def log(m):
    line = f'{time.strftime("%H:%M:%S")}  {m}'
    print(line, flush=True); open(OUT / 'rerun.log', 'a').write(line + '\n')

def tie_aware_threshold(benign: np.ndarray, budget: float):
    """Lowest threshold whose realised false-alert rate still satisfies the budget.

    Returns (threshold, realised_rate, attainable). When even the highest observed
    benign score realises more than the budget the point is unattainable with a
    deterministic threshold; the run says so rather than silently exceeding it.
    """
    u = np.unique(benign)                      # ascending
    cand = np.concatenate([u, [np.nextafter(u[-1], np.inf)]])
    rates = np.array([(benign >= t).mean() for t in cand])
    ok = np.flatnonzero(rates <= budget)
    if ok.size == 0:
        t = cand[-1]
        return float(t), float(rates[-1]), False
    i = ok[0]                                  # lowest threshold that satisfies it
    return float(cand[i]), float(rates[i]), bool(cand[i] <= u[-1])

def run_cell(tr, te, cond, seed, train_family, test_family, dev):
    cols = feature_columns(tr, CONDITIONS[cond])
    m = _model('trees', seed).fit(tr[cols].to_numpy(), tr.label.to_numpy())
    s_te = m.predict_proba(te[cols].to_numpy())[:, 1]
    y = te.label.to_numpy()
    benign_target = s_te[y == 0]
    trb = tr[tr.label == 0]
    benign_source = m.predict_proba(trb[cols].to_numpy())[:, 1]
    att = y == 1
    camps = te.campaign.to_numpy()[att]
    s_att = s_te[att]
    cells, percamp = [], []
    for b in TIGHT_BUDGETS:
        for mode, pool in (('oracle', benign_target), ('source', benign_source)):
            thr, realised_cal, attainable = tie_aware_threshold(pool, b)
            realised_target = float((benign_target >= thr).mean())
            # per campaign: first alerting position in the chronological subsample
            det, pos = [], []
            for c in pd.unique(camps):
                v = s_att[camps == c]
                f = np.flatnonzero(v >= thr)
                det.append(bool(f.size)); pos.append(int(f[0]) + 1 if f.size else -1)
                percamp.append(dict(held_out=dev, condition=cond, seed=seed,
                                    train_family=train_family, test_family=test_family,
                                    false_alert_budget=b, calibration=mode, campaign=c,
                                    rows=int((camps == c).sum()),
                                    detected=bool(f.size),
                                    rows_to_first_alert=int(f[0]) + 1 if f.size else -1))
            det = np.array(det); pos = np.array(pos, dtype=float); pos[pos < 0] = np.nan
            cells.append(dict(held_out=dev, condition=cond, seed=seed,
                train_family=train_family, test_family=test_family,
                false_alert_budget=b, calibration=mode,
                nominal_attainable=attainable,
                realised_rate_on_calibration_pool=realised_cal,
                realised_false_alert_rate_target=realised_target,
                campaign_recall=float(det.mean()), campaigns=int(det.size),
                campaigns_detected=int(det.sum()),
                row_recall=float((s_att >= thr).mean()),
                median_rows_to_detection=float(np.nanmedian(pos)) if det.any() else np.nan,
                detected_by_10=float((np.nan_to_num(pos, nan=np.inf) <= 10).mean()),
                detected_by_100=float((np.nan_to_num(pos, nan=np.inf) <= 100).mean()),
                status='OK'))
    return cells, percamp

def main():
    t0 = time.time()
    if CACHE.exists():
        df = pd.read_pickle(CACHE); log(f'sample from cache: {len(df)} rows')
    else:
        devs = device_dirs(ROOT / 'data_real/nbaiot')
        log(f'{len(devs)} devices; sampling with the original seed 0')
        frames = []
        for d in devs:
            f, _ = load_device(d, 12000, 4000, seed=0)
            frames.append(f); log(f'  {d.name}: {len(f)} rows')
        df = pd.concat(frames, ignore_index=True)
        df.to_pickle(CACHE); log(f'sample cached: {len(df)} rows')
    cells, percamp = [], []
    for seed in (0, 1, 2):
        for trf, tef in (('gafgyt', 'mirai'), ('mirai', 'gafgyt')):
            for dev in sorted(df.device.unique()):
                tr = df[(df.device != dev) & (df.family.isin(['benign', trf]))]
                te = df[(df.device == dev) & (df.family.isin(['benign', tef]))]
                if te[te.label == 1].empty:
                    cells.append(dict(held_out=dev, seed=seed, train_family=trf,
                                      test_family=tef, status=f'SKIPPED: no {tef} captures'))
                    continue
                for cond in CONDITIONS:
                    c, pc = run_cell(tr, te, cond, seed, trf, tef, dev)
                    cells += c; percamp += pc
                log(f'  seed {seed} {trf}->{tef} {dev} done  ({time.time()-t0:.0f}s)')
            # persist after every direction, so an interruption costs one block
            pd.DataFrame(cells).to_csv(OUT / 'hard_task_rerun.csv', index=False)
            pd.DataFrame(percamp).to_csv(OUT / 'per_campaign_detection.csv', index=False)
            log(f'seed {seed} {trf}->{tef} PERSISTED: {len(cells)} cells')
    json.dump(dict(completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   cells=len(cells), per_campaign_rows=len(percamp),
                   elapsed_s=round(time.time() - t0, 1),
                   changes=['tie-aware threshold', 'oracle and source calibration',
                            'realised rate retained', 'per-campaign detection retained']),
              open(OUT / 'rerun_manifest.json', 'w'), indent=1)
    log(f'DONE in {time.time()-t0:.0f}s')

if __name__ == '__main__':
    main()
