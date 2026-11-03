# ed-ecg-machine-reading

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23886807.svg)](https://doi.org/10.5281/zenodo.23886807)

Analysis code and aggregate logs for the manuscript *Averaged training runs and a convolutional
network's standing against the stored emergency electrocardiogram report* (in preparation).

The study uses the ed-bwh-ecg dataset (version 1) from Nightingale Open Science: emergency department
visits at Brigham and Women's Hospital, 2010–2015, with a 12-lead ECG in the GE MUSE system. It compares
a gradient-boosted model on the stored machine reading (eight measurements, 21 statement flags, age, sex)
with a 12-lead convolutional network on the raw waveform, for a 30-day composite of death or troponin
elevation, and asks how the comparison changes with the number of averaged training runs.

**No data are included.** The data are available from Nightingale Open Science on application
(https://www.ngsci.org) and cannot be redistributed. Set `NGSCI_DATA` to the folder that holds
`ed-bwh-ecg/` (default `C:/Data/Nightingale`). Derived arrays and out-of-fold predictions go to
`ed-bwh-ecg/turunan/` under it, never into this repository. The logs and figures hold aggregate results only.

## Requirements

Tree models, bootstraps and figures: Python 3.12.10 with NumPy 2.5.2, pandas 3.0.5, scikit-learn 1.9.0,
SciPy 1.18.1, matplotlib 3.11.1. Convolutional networks (`02_cnn.py`, `06_cnn12.py`, `10_benih_tambahan.py`
and the scripts that retrain them): a separate GPU environment with PyTorch 2.14.0 (CUDA 13.0),
NumPy 2.5.4, pandas 2.3.3, scikit-learn 1.9.1. Random seeds are set in the scripts.

## Layout

Scripts, logs, and figures sit together in `analysis/`; scripts import `01_garis_dasar.py` from the same
folder. Scripts run in numerical order; each docstring gives its usage and the log it writes
(`python 01_garis_dasar.py > garis_dasar.log`). Comments and log text are in Indonesian (*kohort* = cohort,
*benih* = seed/run, *lipatan* = fold, *selisih* = difference, *ansambel* = ensemble, *kematian* = death,
*subkelompok* = subgroup, *kontras* = contrast). Model labels in the code: M1g = Report, M1mg = Measurements,
M4 = CNN-12, M5 = Report+CNN-12. Docstrings mention working notes (`Artikel_EDBWH_Rencana.md`,
`Hasil_Uji_EDBWH.md`) that are not part of this repository.

The ensemble-size curve (Figure 2) is `15_ukuran_ansambel.py`, drawn by `16_gambar_kurva.py`; the paired
contrasts are `14_kontras.py`. The calibration plot and the full subgroup tables are in `gambar/` and
`subkelompok.log`.

## Replication on PTB-XL+ and literature audit

Pre-registered on 2 November 2026; the text, unchanged, is `prapendaftaran_ptbxl_audit.md` (Indonesian).

- Replication: PTB-XL 1.0.3 and PTB-XL+ 1.0.1 from PhysioNet (open, CC BY 4.0). Set `PTBXL_DATA` to a folder
  holding `xl/` and `plus/` (default `C:/Data/PhysioNet/ptbxl`); derived arrays go to `turunan/` under it.
  `18_ptbxl_siapkan.py` builds the cohort, Report features and waveforms; `19_ptbxl_cnn.py <target> <runs>` trains
  CNN-12 (GPU environment); `20_ptbxl_report.py <target> <runs> [report|semua]` fits Report; `21_ptbxl_kurva.py`
  draws the ensemble-size curve. Targets MI (25 runs), STTC, CD, HYP (10 runs each); `semua` = all 784 12SL columns.
- Post hoc, not in the pre-registration (added 3 November 2026): `22_ptbxl_unig.py` writes Report features from a
  second manufacturer's program, Uni-G (University of Glasgow), in PTB-XL+: `unig8` (eight global measurements, age,
  sex), `unig_semua` (all 748 Uni-G columns, age, sex), and the same-vendor anchor `sl8` (eight 12SL global
  measurements, age, sex). Run `20_ptbxl_report.py <target> 10 <set>` then `21_ptbxl_kurva.py <target> 10 <set>`.
- Audit: `audit_pustaka/` holds the PubMed query, all PMIDs, the seeded random order, the screening log, the coding
  sheet for 60 studies, and `17_audit_ringkas.py` (Wilson intervals) with its log. Bibliographic data only.

## Licence

MIT. See `LICENSE`.
