"""Download the N-BaIoT release from the UCI Machine Learning Repository and unpack it.

Writes data_real/nbaiot/<device>/benign_traffic.csv and
data_real/nbaiot/<device>/{gafgyt,mirai}_attacks/*.csv, the layout src/devicepulse/data.py
reads. The archive digest is checked against the one recorded for this study.

The per-device attack captures are RAR archives inside the zip; they are unpacked with
bsdtar (libarchive) or unar, one of which must be on PATH.

Usage:  python get_data.py
"""
import hashlib, shutil, subprocess, sys, urllib.request, zipfile
from pathlib import Path

URL = 'https://archive.ics.uci.edu/static/public/442/detection+of+iot+botnet+attacks+n+baiot.zip'
SHA256 = '64929678b081d8e579a8d7c488cf11cc588403f282d5fb065b4156edbd55de9b'
ROOT = Path(__file__).resolve().parent / 'data_real'
ZIP, OUT = ROOT / 'nbaiot.zip', ROOT / 'nbaiot'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ROOT.mkdir(exist_ok=True)
    if not ZIP.exists():
        print('downloading', URL, '(about 1.8 GB)')
        urllib.request.urlretrieve(URL, ZIP)
    got = sha256(ZIP)
    if got != SHA256:
        sys.exit(f'digest mismatch: {got} != {SHA256}')
    print('digest verified')
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(OUT)
    tool = shutil.which('bsdtar') or shutil.which('unar')
    if tool is None:
        sys.exit('install bsdtar (libarchive) or unar to unpack the .rar attack archives')
    for rar in sorted(OUT.glob('*/*.rar')):
        dest = rar.with_suffix('')
        dest.mkdir(exist_ok=True)
        cmd = [tool, '-xf', str(rar), '-C', str(dest)] if tool.endswith('bsdtar') \
            else [tool, '-q', '-f', '-D', '-o', str(dest), str(rar)]
        subprocess.run(cmd, check=True)
        print('unpacked', rar.relative_to(OUT))
    print('done:', sum(1 for _ in OUT.glob('*/**/*.csv')), 'csv files under', OUT)


if __name__ == '__main__':
    main()
