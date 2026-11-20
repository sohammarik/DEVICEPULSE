"""DEVICEPULSE: does causal device history generalise to unseen devices or memorise them?

The paper's own risk note says history can encode device identity. That is testable and
the test matters: the standard evaluation protocol in device-traffic intrusion detection
mixes the same devices through train and test, so a model that recognises a device rather
than an attack scores well.

Three things are measured together:
  1. detection under the leaky protocol  (random split, same devices both sides)
  2. detection under the honest protocol (leave-one-device-out, device unseen in training)
  3. whether device identity is itself recoverable from the history features

All at a matched grouped false-alert budget, scored per attack campaign rather than per
packet-statistic row, so one long campaign counts once.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import CURRENT_ONLY, HISTORY, feature_columns

CONDITIONS = {'current window only': CURRENT_ONLY,
              'current window plus causal history': CURRENT_ONLY + HISTORY}


def _model(name, seed):
    if name == 'trees':
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                              max_depth=5, random_state=seed)
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=3000, C=1.0, random_state=seed))


def budget_threshold(benign_scores: np.ndarray, false_alert_budget: float) -> float:
    """Threshold set so the benign false-alert rate equals the budget, on benign data only."""
    return float(np.quantile(benign_scores, 1 - false_alert_budget))


def campaign_recall(scores, labels, campaigns, thr) -> tuple[float, int, int]:
    """A campaign counts once: detected if any of its rows crosses the threshold."""
    fired = scores >= thr
    att = labels == 1
    camps = pd.Series(campaigns)[att]
    hit = pd.Series(fired)[att].groupby(camps.to_numpy()).any()
    return (float(hit.mean()), int(hit.sum()), int(len(hit)))


def evaluate(train, test, cond, model='trees', budgets=(0.001, 0.005, 0.01, 0.05),
             seed=0, protocol='', held_out='') -> list[dict]:
    cols = feature_columns(train, CONDITIONS[cond])
    m = _model(model, seed).fit(train[cols].to_numpy(), train.label.to_numpy())
    s = m.predict_proba(test[cols].to_numpy())[:, 1]
    y = test.label.to_numpy()
    benign = s[y == 0]
    rows = []
    for b in budgets:
        thr = budget_threshold(benign, b)
        rec, hit, tot = campaign_recall(s, y, test.campaign.to_numpy(), thr)
        rows.append(dict(protocol=protocol, held_out=held_out, condition=cond,
                         model=model, n_features=len(cols), false_alert_budget=b,
                         realised_false_alert_rate=float((benign >= thr).mean()),
                         campaign_recall=rec, campaigns_detected=hit, campaigns=tot,
                         row_recall=float((s[y == 1] >= thr).mean()),
                         n_train=len(train), n_test=len(test), seed=seed))
    return rows


def leave_one_device_out(df, model='trees', seed=0) -> pd.DataFrame:
    out = []
    for dev in sorted(df.device.unique()):
        tr, te = df[df.device != dev], df[df.device == dev]
        for cond in CONDITIONS:
            out += evaluate(tr, te, cond, model, seed=seed,
                            protocol='leave-one-device-out', held_out=dev)
    return pd.DataFrame(out)


def random_split(df, model='trees', seed=0, test_size=0.3) -> pd.DataFrame:
    tr, te = train_test_split(df, test_size=test_size, random_state=seed,
                              stratify=df.label)
    out = []
    for cond in CONDITIONS:
        out += evaluate(tr, te, cond, model, seed=seed,
                        protocol='random split (devices shared)', held_out='none')
    return pd.DataFrame(out)


def device_identity_probe(df, seed=0) -> pd.DataFrame:
    """Can the device be recovered from the features alone, on benign traffic only?

    Benign only, so the probe cannot succeed by recognising an attack. Chance accuracy is
    one over the number of devices.
    """
    b = df[df.label == 0]
    rows = []
    for cond, decays in CONDITIONS.items():
        cols = feature_columns(b, decays)
        X, y = b[cols].to_numpy(), b.device.to_numpy()
        tr, te, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed,
                                            stratify=y)
        m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1,
                                           random_state=seed).fit(tr, ytr)
        acc = accuracy_score(yte, m.predict(te))
        rows.append(dict(condition=cond, n_features=len(cols),
                         device_identification_accuracy=float(acc),
                         chance=1.0 / b.device.nunique(),
                         n_devices=int(b.device.nunique()), n_benign_rows=len(b)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------
# Hard task. The standard protocol saturates: campaign recall is 1.000 everywhere and
# row recall exceeds 0.98, so the benchmark cannot separate methods. Three things are
# made harder at once, each for a stated reason.
#   unseen device   the held-out device never appears in training
#   unseen family   training sees BASHLITE only; the test is Mirai
#   timeliness      scored by how many attack rows pass before the first alert
# --------------------------------------------------------------------------------
TIGHT_BUDGETS = (1e-4, 3e-4, 1e-3, 3e-3)


def time_to_detection(scores, campaigns, thr) -> pd.DataFrame:
    """Rows observed within a campaign before its first alert. Censored if never."""
    out = []
    s = pd.Series(scores)
    for camp, idx in pd.Series(campaigns).groupby(pd.Series(campaigns)).groups.items():
        v = s.loc[idx].to_numpy()
        fired = np.flatnonzero(v >= thr)
        out.append(dict(campaign=camp, rows=len(v),
                        rows_to_first_alert=int(fired[0]) + 1 if len(fired) else np.nan,
                        detected=bool(len(fired))))
    return pd.DataFrame(out)


def hard_task(df, model='trees', seed=0,
              train_family='gafgyt', test_family='mirai') -> pd.DataFrame:
    """Leave-one-device-out, train on one attack family, test on the other."""
    rows = []
    for dev in sorted(df.device.unique()):
        tr = df[(df.device != dev) & (df.family.isin(['benign', train_family]))]
        te = df[(df.device == dev) & (df.family.isin(['benign', test_family]))]
        if te[te.label == 1].empty:
            rows.append(dict(held_out=dev, condition='n/a', status=
                             f'SKIPPED: no {test_family} captures for this device'))
            continue
        for cond in CONDITIONS:
            cols = feature_columns(tr, CONDITIONS[cond])
            m = _model(model, seed).fit(tr[cols].to_numpy(), tr.label.to_numpy())
            s = m.predict_proba(te[cols].to_numpy())[:, 1]
            y = te.label.to_numpy()
            benign = s[y == 0]
            for b in TIGHT_BUDGETS:
                thr = budget_threshold(benign, b)
                att = y == 1
                ttd = time_to_detection(s[att], te.campaign.to_numpy()[att], thr)
                rows.append(dict(held_out=dev, condition=cond, model=model, seed=seed,
                                 train_family=train_family, test_family=test_family,
                                 false_alert_budget=b,
                                 realised_false_alert_rate=float((benign >= thr).mean()),
                                 campaign_recall=float(ttd.detected.mean()),
                                 campaigns=int(len(ttd)),
                                 median_rows_to_detection=float(
                                     ttd.loc[ttd.detected, 'rows_to_first_alert'].median())
                                 if ttd.detected.any() else np.nan,
                                 mean_rows_to_detection=float(
                                     ttd.loc[ttd.detected, 'rows_to_first_alert'].mean())
                                 if ttd.detected.any() else np.nan,
                                 row_recall=float((s[att] >= thr).mean()),
                                 status='OK'))
    return pd.DataFrame(rows)
