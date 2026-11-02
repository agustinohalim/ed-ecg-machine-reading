"""Nightingale ed-bwh-ecg — bootstrap 25 benih tanpa pasien pencilan, tanpa melatih apa pun.

Menjawab telaah 31 Okt 2026 (Telaah/2026-10-31-EDBWH-Kinetik): selisih utama diulang tanpa pasien dengan kejadian
terbanyak, dengan cara 10_benih_tambahan.py (pasien sebagai unit, lima benih berbeda dari 25, 1.000 ulangan, benih
bootstrap 20261031). Prediksi OOF tetap dari model yang melihat pasien itu saat dilatih; model tidak dipasang ulang.

Pakai: python 12_tanpa_pencilan.py > tanpa_pencilan.log
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

D = os.path.join(os.environ.get('NGSCI_DATA', 'C:/Data/Nightingale'), 'ed-bwh-ecg')
T = os.path.join(D, 'turunan')
S = 'macetrop_pos_or_death_030'


def ci(v, q=(2.5, 97.5)):
    a, b = np.percentile(v, q)
    return f'[{a:+.3f}; {b:+.3f}]'


def main():
    k = pd.read_csv(os.path.join(T, 'kohort.csv'))
    y, g = k[S].values, k['patient_ngsci_id'].values
    top = pd.DataFrame({'g': g, 'y': y}).groupby('g').y.sum().idxmax()
    m = g != top
    print(f'dikeluarkan {(~m).sum()} kunjungan, {y[~m].sum()} kejadian; sisa {m.sum()} kunjungan, {y[m].sum()} kejadian')
    PP = {n: np.load(os.path.join(T, f'oof_{S}_{n}_25.npy'))[:, m] for n in ('M1g', 'M1mg', 'M4', 'M5')}
    y, g = y[m], g[m]
    nb = PP['M4'].shape[0]
    ug, inv = np.unique(g, return_inverse=True)
    urut = np.argsort(inv, kind='stable')
    batas = np.searchsorted(inv[urut], np.arange(len(ug) + 1))
    baris = [urut[batas[j]:batas[j + 1]] for j in range(len(ug))]
    rng = np.random.default_rng(20261031)
    idx = [np.concatenate([baris[j] for j in rng.integers(0, len(baris), len(baris))]) for _ in range(1000)]
    pilih = [rng.choice(nb, 5, replace=False) for _ in idx]
    satu = rng.integers(0, nb, len(idx))
    for a, b in [('M4', 'M1g'), ('M5', 'M1g'), ('M4', 'M1mg'), ('M5', 'M1mg')]:
        d0 = roc_auc_score(y, PP[a][:5].mean(0)) - roc_auc_score(y, PP[b][:5].mean(0))
        ans = [roc_auc_score(y[i], PP[a][s].mean(0)[i]) - roc_auc_score(y[i], PP[b][s].mean(0)[i])
               for i, s in zip(idx, pilih)]
        sat = [roc_auc_score(y[i], PP[a][s][i]) - roc_auc_score(y[i], PP[b][s][i]) for i, s in zip(idx, satu)]
        d1 = np.mean([roc_auc_score(y, PP[a][s]) - roc_auc_score(y, PP[b][s]) for s in range(nb)])
        print(f'{a} − {b}: ansambel 0–4 {d0:+.3f}, pasien + benih {ci(ans)} CI90 {ci(ans, (5, 95))}; '
              f'satu jaringan {d1:+.3f} {ci(sat)} CI90 {ci(sat, (5, 95))}', flush=True)


if __name__ == '__main__':
    main()
