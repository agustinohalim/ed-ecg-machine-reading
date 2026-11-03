"""PTB-XL+ — model Report (HistGB atas bacaan mesin 12SL) per putaran, lipatan sama dengan 19_ptbxl_cnn.py.

Pengaturan M1g ED-BWH: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0); nilai hilang
ditangani HistGB sendiri. Putaran b = StratifiedGroupKFold(5, shuffle, random_state=b), grup = pasien, jadi Report
berubah antarputaran hanya lewat pembagian lipatan, seperti di ED-BWH.

  ciri 'report' = delapan ukuran global + pernyataan >= 1 % + usia + jenis kelamin (ciri_report.csv)
  ciri 'semua'  = semua kolom 12SL + usia + jenis kelamin (ciri_12sl_semua.csv), analisis kepekaan
  ciri 'unig8', 'unig_semua', 'sl8' = berkas dari 22_ptbxl_unig.py (analisis pasca-hasil, Hasil bagian C)

Pakai: python 20_ptbxl_report.py <sasaran> <jumlah putaran> [report|semua] > ptbxl_report_<sasaran>.log
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

T = os.path.join(os.environ.get('PTBXL_DATA', 'C:/Data/PhysioNet/ptbxl'), 'turunan')


def main():
    s, n = sys.argv[1], int(sys.argv[2])
    jenis = sys.argv[3] if len(sys.argv) > 3 else 'report'
    d = pd.read_csv(os.path.join(T, 'kohort.csv'))
    berkas = {'report': 'ciri_report.csv', 'semua': 'ciri_12sl_semua.csv'}.get(jenis, f'ciri_{jenis}.csv')
    X = pd.read_csv(os.path.join(T, berkas))
    assert (X.ecg_id.values == d.ecg_id.values).all()
    X = X.drop(columns='ecg_id').values.astype(float)
    y, g = d[s].values.astype(int), d['patient_id'].values
    print(f'{s} {jenis}: {len(y)} rekaman, {y.sum()} positif, {X.shape[1]} ciri', flush=True)
    P = np.zeros((n, len(y)))
    for b in range(n):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=b).split(X, y, g):
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(X[tr], y[tr])
            P[b, te] = m.predict_proba(X[te])[:, 1]
        print(f'  benih {b}: AUROC {roc_auc_score(y, P[b]):.3f}', flush=True)
    np.save(os.path.join(T, f'oof_{s}_{jenis}_{n}.npy'), P)
    a = [roc_auc_score(y, p) for p in P]
    print(f'rerata AUROC {np.mean(a):.3f} (SD {np.std(a):.3f}); ansambel {n}: {roc_auc_score(y, P.mean(0)):.3f}')


if __name__ == '__main__':
    main()
