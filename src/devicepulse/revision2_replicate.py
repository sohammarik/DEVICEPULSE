"""Exact published hard-protocol configuration (rerun_review.run_cell, full training
benign rows, oracle and in-sample source calibration) on the revision-2 sample, 3 seeds,
both directions. Separates a change of row sample from the change of training rows that
the revision-2 calibration holdout introduces.

Usage:  PYTHONHASHSEED=0 python src/devicepulse/revision2_replicate.py
"""
import sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from devicepulse.experiment import CONDITIONS
from devicepulse.rerun_review import run_cell
from devicepulse import rerun_review
from devicepulse.revision2 import sample, OUT, log, tie_aware_threshold
# identical results (checked against the reference on 2,100 cases), sorted search instead
# of a rescan per candidate
rerun_review.tie_aware_threshold = tie_aware_threshold

df = sample(); cells = []; t0 = time.time()
for seed in (0, 1, 2):
    for trf, tef in (('gafgyt', 'mirai'), ('mirai', 'gafgyt')):
        for dev in sorted(df.device.unique()):
            tr = df[(df.device != dev) & (df.family.isin(['benign', trf]))]
            te = df[(df.device == dev) & (df.family.isin(['benign', tef]))]
            if te[te.label == 1].empty:
                continue
            for cond in CONDITIONS:
                cells += run_cell(tr, te, cond, seed, trf, tef, dev)[0]
        log(f'[replicate] s{seed} {trf}->{tef} ({time.time()-t0:.0f}s)')
        pd.DataFrame(cells).to_csv(OUT / 'replicate_published_config.csv', index=False)
log('[replicate] DONE')
