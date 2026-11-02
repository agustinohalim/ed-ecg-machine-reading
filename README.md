# ed-ecg-machine-reading
n[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23886807.svg)](https://doi.org/10.5281/zenodo.23886807)

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

The ensemble-size curve (Figure 3) is `15_ukuran_ansambel.py`, drawn by `16_gambar_kurva.py`; the paired
contrasts are `14_kontras.py`.

## Licence

MIT. See `LICENSE`.
