"""Nightingale ed-bwh-ecg — kurva ukuran ansambel atas telaah ketujuh, tanpa melatih apa pun.

Telaah 2026-11-01b-EDBWH-Kinetik: kontras perataan hanya k = 1 lawan k = 5 padahal ada 25 jaringan; tidak diulang tanpa
pasien pencilan dan pada kunjungan pertama; AUROC kematian belum diperiksa di dalam lipatan. Benih bootstrap 20261103.

  1. Kurva k = 1, 2, 3, 5, 7, 10, 15, 25. Tiap ulangan bootstrap menarik pasien dan satu permutasi 25 benih; k benih
     pertama permutasi itu dipakai untuk M4 dan M1g sekaligus, sehingga ansambel bersarang dan berpasangan menurut benih.
     Titik = rerata atas semua kunjungan: k = 1 tepat atas 25 benih, k = 25 satu pilihan, lainnya 200 pilihan acak.
     Dilaporkan AUROC M4 dan M1g, M4 − M1g, dan (M4 − M1g)_k − (M4 − M1g)_1.
  2. Kontras ansambel lima − satu jaringan dengan cara 14_kontras.py: semua kunjungan sudah di sana; di sini tanpa
     pasien pencilan dan pada kunjungan pertama.
  3. AUROC di dalam lipatan untuk model kematian yang dipasang ulang, benih 0–4.

Pakai: PYTHONIOENCODING=utf-8 python 15_ukuran_ansambel.py > ukuran_ansambel.log
"""
import importlib.util
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

D = os.path.join(os.environ.get('NGSCI_DATA', 'C:/Data/Nightingale'), 'ed-bwh-ecg')
T = os.path.join(D, 'turunan')
S = 'macetrop_pos_or_death_030'
K = (1, 2, 3, 5, 7, 10, 15, 25)
spek = importlib.util.spec_from_file_location('gd', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                 '01_garis_dasar.py'))
gd = importlib.util.module_from_spec(spek)
spek.loader.exec_module(gd)


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


def kontras(y, g, PP, rng, label):
    idx = sampel(g, rng)
    pilih = [rng.choice(25, 5, replace=False) for _ in idx]
    satu = rng.integers(0, 25, len(idx))
    d5 = auc(y, PP['M4'][:5].mean(0)) - auc(y, PP['M1g'][:5].mean(0))
    d1 = np.mean([auc(y, PP['M4'][s]) - auc(y, PP['M1g'][s]) for s in range(25)])
    v = [(auc(y, PP['M4'][s].mean(0), i) - auc(y, PP['M1g'][s].mean(0), i))
         - (auc(y, PP['M4'][t], i) - auc(y, PP['M1g'][t], i)) for i, s, t in zip(idx, pilih, satu)]
    print(f'  {label}: {d5 - d1:+.3f} CI95 {ci(v)} CI90 {ci(v, (5, 95))} (ansambel {d5:+.3f}, satu {d1:+.3f})', flush=True)


def main():
    d = gd.kohort()
    y, g = d[S].values, d['patient_ngsci_id'].values
    mati = d['death_030_day'].values
    PP = {n: np.load(os.path.join(T, f'oof_{S}_{n}_25.npy')) for n in ('M1g', 'M4')}
    rng = np.random.default_rng(20261103)

    print('== 1. Kurva ukuran ansambel, semua kunjungan')
    idx = sampel(g, rng)
    perm = [rng.permutation(25) for _ in idx]
    B = {k: {'M4': [], 'M1g': []} for k in K}
    for i, p in zip(idx, perm):
        for k in K:
            for n in ('M4', 'M1g'):
                B[k][n].append(auc(y, PP[n][p[:k]].mean(0), i))
    d1b = np.array(B[1]['M4']) - np.array(B[1]['M1g'])
    for k in K:
        t = {n: [] for n in ('M4', 'M1g')}
        pil = [[j] for j in range(25)] if k == 1 else [range(25)] if k == 25 else             [rng.choice(25, k, replace=False) for _ in range(200)]
        for s in pil:
            s = list(s)
            for n in t:
                t[n].append(auc(y, PP[n][s].mean(0)))
        a, b = np.array(t['M4']), np.array(t['M1g'])
        d1t = np.mean([auc(y, PP['M4'][s]) - auc(y, PP['M1g'][s]) for s in range(25)])
        db = np.array(B[k]['M4']) - np.array(B[k]['M1g'])
        print(f'  k={k:2d}: M4 {a.mean():.3f} M1g {b.mean():.3f} selisih {(a - b).mean():+.3f} {ci(db)} '
              f'CI90 {ci(db, (5, 95))}; lawan k=1 {(a - b).mean() - d1t:+.3f} {ci(db - d1b)} '
              f'CI90 {ci(db - d1b, (5, 95))}', flush=True)

    print('== 2. Kontras ansambel lima − satu jaringan')
    top = pd.DataFrame({'g': g, 'y': y}).groupby('g').y.sum().idxmax()
    m = g != top
    kontras(y[m], g[m], {n: P[:, m] for n, P in PP.items()}, rng, 'tanpa pasien')
    u = (pd.DataFrame({'g': g, 't': pd.to_datetime(d['start_datetime'])}).reset_index()
         .sort_values('t').drop_duplicates('g')['index'].sort_values().values)
    kontras(y[u], g[u], {n: P[:, u] for n, P in PP.items()}, rng, 'kunjungan pertama')

    print('== 3. Kematian, model dipasang ulang, AUROC gabungan lawan di dalam lipatan, benih 0–4')
    PK = {n: np.load(os.path.join(T, f'oof_death_030_day_{n}.npy')) for n in ('M1g', 'M4')}
    r = {n: [] for n in PK}
    for b in range(5):
        lip = list(StratifiedGroupKFold(5, shuffle=True, random_state=b).split(np.zeros(len(mati)), mati, g))
        for n, P in PK.items():
            r[n].append((auc(mati, P[b]), np.mean([auc(mati[te], P[b][te]) for _, te in lip])))
    for n, v in r.items():
        a = np.array(v)
        print(f'  {n}: gabungan {a[:, 0].mean():.3f}, di dalam lipatan {a[:, 1].mean():.3f}')
    a, b = np.array(r['M4']), np.array(r['M1g'])
    print(f'  M4 − M1g satu jaringan: gabungan {(a[:, 0] - b[:, 0]).mean():+.3f}, '
          f'di dalam lipatan {(a[:, 1] - b[:, 1]).mean():+.3f}')


if __name__ == '__main__':
    main()
