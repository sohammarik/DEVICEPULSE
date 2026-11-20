"""Real N-BaIoT traffic statistics: loading, device structure and provenance.

Nine commercial IoT devices, real benign traffic and real BASHLITE (gafgyt) and Mirai
infections. Features are the dataset's own damped incremental statistics over five decay
factors; nothing is engineered here beyond selecting which decay windows a condition is
allowed to see.

Decay factors are lambda in {5, 3, 1, 0.1, 0.01}. A larger lambda decays faster, so L5 is
the most nearly instantaneous view and L0.01 carries the longest memory. The paper's
contrast between a current-window detector and one with causal device history is
therefore available directly in the dataset's own feature structure.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

DECAYS = ('L5', 'L3', 'L1', 'L0.1', 'L0.01')
CURRENT_ONLY = ('L5',)                       # nearest to instantaneous
HISTORY = ('L3', 'L1', 'L0.1', 'L0.01')      # the device's causal history


def device_dirs(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and (p / 'benign_traffic.csv').exists())


def _sha(p: Path, limit: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with p.open('rb') as fh:
        h.update(fh.read(limit))
    return h.hexdigest()


def load_device(dev: Path, n_benign: int, n_attack: int, seed: int = 0):
    """Stratified sample of one device's traffic. Returns frame and a provenance record."""
    rng = np.random.default_rng(seed + abs(hash(dev.name)) % 10_000)
    frames, prov = [], []

    def take(path: Path, label: int, family: str, campaign: str, n: int):
        total = sum(1 for _ in path.open('r', encoding='utf-8')) - 1
        keep = min(n, total)
        skip = sorted(rng.choice(np.arange(1, total + 1), size=total - keep,
                                 replace=False)) if keep < total else None
        df = pd.read_csv(path, skiprows=skip)
        df['label'] = label
        df['family'] = family
        df['campaign'] = f'{dev.name}|{family}|{campaign}'
        df['device'] = dev.name
        frames.append(df)
        prov.append(dict(device=dev.name, file=str(path.relative_to(dev.parent)),
                         rows_total=total, rows_sampled=len(df),
                         label=label, family=family, campaign=campaign,
                         sha256_first16mb=_sha(path)))

    take(dev / 'benign_traffic.csv', 0, 'benign', 'benign', n_benign)
    for fam in ('gafgyt_attacks', 'mirai_attacks'):
        d = dev / fam
        if not d.exists():
            continue
        for f in sorted(d.glob('*.csv')):
            take(f, 1, fam.replace('_attacks', ''), f.stem, n_attack)
    return pd.concat(frames, ignore_index=True), prov


def feature_columns(df: pd.DataFrame, decays) -> list[str]:
    """Columns whose decay suffix is in the allowed set, matched on exact token."""
    out = []
    for c in df.columns:
        if c in ('label', 'family', 'campaign', 'device'):
            continue
        parts = c.split('_')
        tok = next((p for p in parts if p in DECAYS), None)
        if tok in decays:
            out.append(c)
    return out
