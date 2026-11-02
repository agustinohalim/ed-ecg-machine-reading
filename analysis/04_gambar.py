"""Nightingale ed-bwh-ecg — Gambar 1 (ROC dan PR) dan Gambar 2 (kalibrasi) untuk naskah.

Prediksi = OOF M0/M1g/M2/M4/M5 dirata-ratakan (M3 hanya di tabel, sejak 28 Okt) atas 5 benih (sama dengan angka utama di Hasil_Uji_EDBWH.md).
Kalibrasi: kunjungan dibagi 10 desil probabilitas terduga; titik = rerata terduga lawan proporsi
teramati per desil, garis bantu y = x. Gambar hanya memuat kurva agregat, jadi boleh masuk repo
(folder gambar/); data per kunjungan tetap di folder turunan.

Pakai: python 04_gambar.py [sasaran]   (baku macetrop_pos_or_death_030)
"""
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_curve, roc_auc_score,
                             roc_curve)

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

D = os.path.join(os.environ.get('NGSCI_DATA', 'C:/Data/Nightingale'), 'ed-bwh-ecg')
T = os.path.join(D, 'turunan')
G = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gambar')
NAMA = {'M0': 'Age-sex', 'M1g': 'Report', 'M2': 'CNN-3', 'M4': 'CNN-12', 'M5': 'Report+CNN-12'}
WARNA = {'M0': '#999999', 'M1g': '#1f77b4', 'M2': '#ff9896', 'M4': '#d62728', 'M5': '#2ca02c'}


def main():
    s = sys.argv[1] if len(sys.argv) > 1 else 'macetrop_pos_or_death_030'
    os.makedirs(G, exist_ok=True)
    y = pd.read_csv(os.path.join(T, 'kohort.csv'))[s].values
    P = {n: np.load(os.path.join(T, f'oof_{s}_{n}.npy')).mean(0) for n in NAMA
         if os.path.exists(os.path.join(T, f'oof_{s}_{n}.npy'))}
    plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans'})

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for n, p in P.items():
        fpr, tpr, _ = roc_curve(y, p)
        a.plot(fpr, tpr, color=WARNA[n], lw=1.3, label=f'{NAMA[n]} ({roc_auc_score(y, p):.3f})')
        pr, rc, _ = precision_recall_curve(y, p)
        b.plot(rc, pr, color=WARNA[n], lw=1.3, label=f'{NAMA[n]} ({average_precision_score(y, p):.3f})')
    a.plot([0, 1], [0, 1], ':', color='k', lw=0.8)
    b.axhline(y.mean(), ls=':', color='k', lw=0.8)
    a.set(xlabel='1 − specificity', ylabel='Sensitivity', title='(a) ROC', xlim=(0, 1), ylim=(0, 1))
    b.set(xlabel='Recall', ylabel='Precision', title='(b) Precision–recall', xlim=(0, 1), ylim=(0, 1))
    a.legend(loc='lower right', fontsize=6.5, title='AUROC', title_fontsize=7)
    b.legend(loc='upper right', fontsize=6.5, title='AUPRC', title_fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(G, f'gambar1_roc_pr_{s}.png'), dpi=300)

    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    puncak = 0
    for n, p in P.items():
        desil = pd.qcut(p, 10, labels=False, duplicates='drop')
        t = pd.DataFrame({'p': p, 'y': y, 'k': desil}).groupby('k').mean()
        ax.plot(t.p, t.y, 'o-', color=WARNA[n], ms=3, lw=1, label=NAMA[n])
        puncak = max(puncak, t.p.max(), t.y.max())
    lim = min(1, puncak * 1.1)
    ax.plot([0, lim], [0, lim], ':', color='k', lw=0.8)
    ax.set(xlabel='Predicted risk (decile mean)', ylabel='Observed event rate', xlim=(0, lim),
           ylim=(0, lim), title='Calibration by decile')
    ax.legend(fontsize=6.5, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(G, f'gambar2_kalibrasi_{s}.png'), dpi=300)
    print(f'{s}: {len(P)} model, gambar di {G}')


if __name__ == '__main__':
    main()
