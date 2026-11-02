"""Nightingale ed-bwh-ecg — dua angka tambahan untuk revisi Kinetik, tanpa melatih apa pun.

  1. Selisih M5 − M1mg (gabungan lawan ukuran mesin saja) dengan cara 10_benih_tambahan.py: 1.000 ulangan,
     pasien sebagai unit, lima benih berbeda dari 25, benih bootstrap 20261031.
  2. Jumlah kematian dalam 10 % pasien kalibrasi tiap lipatan M4 (rumus pembagian sama dengan 06_cnn12.py).

Pakai: python 11_tambahan_kinetik.py > tambahan_kinetik.log
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

D = os.path.join(os.environ.get('NGSCI_DATA', 'C:/Data/Nightingale'), 'ed-bwh-ecg')
T = os.path.join(D, 'turunan')
S = 'macetrop_pos_or_death_030'


def ci(v, q=(2.5, 97.5)):
    a, b = np.percentile(v, q)
    return f'[{a:+.3f}; {b:+.3f}]'


def main():
    k = pd.read_csv(os.path.join(T, 'kohort.csv'))
    y, g = k[S].values, k['patient_ngsci_id'].values
    A, B = (np.load(os.path.join(T, f'oof_{S}_{n}_25.npy')) for n in ('M5', 'M1mg'))
    nb = A.shape[0]
    ug, inv = np.unique(g, return_inverse=True)
    urut = np.argsort(inv, kind='stable')
    batas = np.searchsorted(inv[urut], np.arange(len(ug) + 1))
    baris = [urut[batas[j]:batas[j + 1]] for j in range(len(ug))]
    rng = np.random.default_rng(20261031)
    idx = [np.concatenate([baris[j] for j in rng.integers(0, len(baris), len(baris))]) for _ in range(1000)]
    idx = [i for i in idx if y[i].min() != y[i].max()]
    pilih = [rng.choice(nb, 5, replace=False) for _ in idx]
    ans = [roc_auc_score(y[i], A[s].mean(0)[i]) - roc_auc_score(y[i], B[s].mean(0)[i]) for i, s in zip(idx, pilih)]
    d0 = roc_auc_score(y, A[:5].mean(0)) - roc_auc_score(y, B[:5].mean(0))
    print(f'M5 − M1mg: ansambel benih 0–4 {d0:+.3f}; pasien + benih {ci(ans)} CI90 {ci(ans, (5, 95))}')

    m = k['death_030_day'].values
    n = []
    Z = np.zeros((len(m), 1))
    for b in range(5):
        for j, (tr, _) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=b).split(Z, m, g)):
            r = np.random.default_rng(1000 * b + j)
            u = np.unique(g[tr])
            cal_g = set(r.choice(u, len(u) // 10, replace=False))
            n.append(int(m[tr][[x in cal_g for x in g[tr]]].sum()))
    print(f'kematian dalam subset kalibrasi per lipatan: median {int(np.median(n))}, rentang {min(n)}–{max(n)}')


if __name__ == '__main__':
    main()
