# DPulse: code and results

Code and result files for the paper *DPulse: A Saturated Benchmark and a Device Fingerprint: Re-examining Causal History in IoT Attack Detection* (S. Marik, A. Chakraborty, S. Chakraborty, S. Dhar and S. Sinha).

The study re-examines evaluation practice for IoT intrusion detection on the N-BaIoT release. It runs:

- the standard protocols (random split, time-ordered split, leave-one-device-out);
- a device-identification probe on benign traffic, with each decay factor tested alone at equal dimension;
- a hard protocol that holds out a device and an attack family together, under two calibration modes in which the alert threshold and the realised false-alert rate are measured on disjoint rows;
- the same hard protocol with an autoencoder trained on benign traffic alone.

## Data

The N-BaIoT release is public: UCI Machine Learning Repository, dataset 442, doi:[10.24432/C5RC8J](https://doi.org/10.24432/C5RC8J). It is not stored here. To download it, verify its digest and unpack it into `data_real/nbaiot/`:

```
python get_data.py
```

This fetches about 1.8 GB and unpacks to about 9.1 GB. The attack captures are RAR archives, so `bsdtar` (libarchive) or `unar` must be installed.

## Setup

Python 3.12 with the pinned packages the results were produced with:

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Running

Every stage reads one shared row sample. The sampler derives each device's seed from Python's string hash, so `PYTHONHASHSEED=0` must be set or the sample, and every number, will differ.

```
export PYTHONHASHSEED=0
python src/devicepulse/revision2.py sample     # draw and cache the row sample
python src/devicepulse/revision2.py standard   # Table I
python src/devicepulse/revision2.py probe      # Table II
python src/devicepulse/revision2.py hard       # Table III, Fig. 1, Fig. 2, Table IV
python src/devicepulse/revision2.py ae         # autoencoder results
python src/devicepulse/revision2.py norm       # per-device standardised features (see below)
python src/devicepulse/revision2_replicate.py  # the first submission's configuration on this sample
python src/devicepulse/figures_revision2.py    # figures and per_device_gain.csv
```

Run the stages one at a time. Each fit uses every core, so running two at once slows both considerably. On a 10-core laptop the full set takes about 1.5 hours, most of it in `hard` and `probe`.

## Where each result comes from

All result files are in `results_revision2/`, and figures with their plotted values are in `figures_revision2/`.

| Paper | File | Selection |
|---|---|---|
| Table I | `standard_protocol.csv` | `calibration == target_disjoint` |
| Table II | `identity_probe.csv` | `standardised == False` |
| Table III, Fig. 1 | `hard_cells.csv` | seeds 0 to 2; `calibration` in `target_disjoint`, `source_heldout` |
| Fig. 2 | `hard_per_campaign.csv` | `target_disjoint`, budget 0.0001, seeds 0 to 2 |
| Table IV | `per_device_gain.csv` | twelve seeds, BASHLITE to Mirai, budget 0.0001 |
| Autoencoder results (Sec. IV) | `ae_cells.csv` | `target_disjoint`, `source_heldout` |
| First-submission configuration on this sample | `replicate_published_config.csv` | `oracle` and `source` calibration as originally run |

Calibration modes in `hard_cells.csv`:

- `target_disjoint`: threshold set on the first half of the held-out device's benign rows in capture order; realised rate measured on the second half.
- `source_heldout`: threshold set on 30 percent of each training device's benign rows, withheld from fitting.
- `target_same`, `source_insample`: the first submission's rules, kept for comparison.

`norm_cells.csv` holds a run on per-device standardised features. It is not used as evidence in the paper, because the standardisation is computed per device and the identity probe still recovers the device from the standardised features (`identity_probe.csv`, `standardised == True`).

`manifest.json` records the Python, scikit-learn, numpy and pandas versions and the platform each stage ran on. `file_provenance.csv` gives every file's row total and sampled count.

## License

The code is released under the MIT License (see `LICENSE`). The N-BaIoT data is not part of this repository and remains under the terms of its UCI release.
