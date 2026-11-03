"""PTB-XL+ — kurva ukuran ansambel CNN-12 lawan Report, cara 15_ukuran_ansambel.py (prapendaftaran 2 Nov 2026).

Tiap ulangan bootstrap (1.000, pasien sebagai unit, benih 20261104) menarik pasien dan satu permutasi putaran; k putaran
pertama dipakai untuk CNN dan Report sekaligus (bersarang, berpasangan menurut putaran). Titik = rerata atas semua
rekaman: k = 1 tepat atas semua putaran, k = n satu pilihan, lainnya 200 pilihan acak. Peringatan kolam terbatas
berlaku: menarik k dari n tanpa pengembalian menyempitkan bagian variasi putaran sekitar (n − k)/(n − 1).

Pakai: python 21_ptbxl_kurva.py <sasaran> <jumlah putaran> [report|semua] > ptbxl_kurva_<sasaran>.log
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

T = os.path.join(os.environ.get('PTBXL_DATA', 'C:/Data/PhysioNet/ptbxl'), 'turunan')


def ci(v, q=(2.5, 97.5)):
    a, b = np.percentile(v, q)
    return f'[{a:+.3f}; {b:+.3f}]'


def sampel(g, rng):
    ug, inv = np.unique(g, return_inverse=True)
    urut = np.argsort(inv, kind='stable')
    batas = np.searchsorted(inv[urut], np.arange(len(ug) + 1))
    baris = [urut[batas[j]:batas[j + 1]] for j in range(len(ug))]
    return [np.concatenate([baris[j] for j in rng.integers(0, len(baris), len(baris))]) for _ in range(1000)]


def auc(y, p, i=None):
    return roc_auc_score(y, p) if i is None else roc_auc_score(y[i], p[i])


def main():
    s, n = sys.argv[1], int(sys.argv[2])
    jenis = sys.argv[3] if len(sys.argv) > 3 else 'report'
    K = [k for k in (1, 2, 3, 5, 7, 10, 15, 25) if k <= n]
    d = pd.read_csv(os.path.join(T, 'kohort.csv'))
    y, g = d[s].values.astype(int), d['patient_id'].values
    PP = {'cnn': np.stack([np.load(os.path.join(T, f'oof_{s}_cnn_benih{b}.npy')) for b in range(n)]),
          'rep': np.load(max(glob.glob(os.path.join(T, f'oof_{s}_{jenis}_*.npy')),
                             key=lambda f: int(f.rsplit('_', 1)[1][:-4])))[:n]}
    sd = {m: np.std([auc(y, p) for p in P]) for m, P in PP.items()}
    print(f'{s} ({jenis}), {n} putaran: {len(y)} rekaman, {y.sum()} positif; SD AUROC antarputaran '
          f'CNN {sd["cnn"]:.3f}, Report {sd["rep"]:.3f}')
    rng = np.random.default_rng(20261104)
    idx = sampel(g, rng)
    perm = [rng.permutation(n) for _ in idx]
    B = {k: {'cnn': [], 'rep': []} for k in K}
    for i, p in zip(idx, perm):
        for k in K:
            for m in B[k]:
                B[k][m].append(auc(y, PP[m][p[:k]].mean(0), i))
    d1b = np.array(B[1]['cnn']) - np.array(B[1]['rep'])
    d1t = np.mean([auc(y, PP['cnn'][j]) - auc(y, PP['rep'][j]) for j in range(n)])
    for k in K:
        pil = [[j] for j in range(n)] if k == 1 else [range(n)] if k == n else \
            [rng.choice(n, k, replace=False) for _ in range(200)]
        a = np.array([auc(y, PP['cnn'][list(j)].mean(0)) for j in pil])
        b = np.array([auc(y, PP['rep'][list(j)].mean(0)) for j in pil])
        db = np.array(B[k]['cnn']) - np.array(B[k]['rep'])
        print(f'  k={k:2d}: CNN {a.mean():.3f} Report {b.mean():.3f} selisih {(a - b).mean():+.3f} {ci(db)} '
              f'CI90 {ci(db, (5, 95))}; lawan k=1 {(a - b).mean() - d1t:+.3f} {ci(db - d1b)} '
              f'CI90 {ci(db - d1b, (5, 95))}', flush=True)


if __name__ == '__main__':
    main()
