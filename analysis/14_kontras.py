"""Nightingale ed-bwh-ecg — kontras berpasangan antar-putusan atas telaah 1 Nov 2026, tanpa melatih apa pun.

Telaah 2026-11-01-EDBWH-Kinetik (kritis, tiga peran): pesan "putusan bergantung pada perataan, pembanding, sasaran"
bertumpu pada label yang berganti, bukan pada selisih yang diuji. Skrip ini menguji selisih itu langsung, dengan
bootstrap yang sama seperti 10_benih_tambahan.py (pasien sebagai unit, 1.000 ulangan, lima benih berbeda dari 25),
benih bootstrap 20261102; "satu jaringan" = satu benih acak per ulangan, pada pasien yang sama.

  1. Perataan: (M4 − M1g) ansambel lima − (M4 − M1g) satu jaringan.
  2. Pembanding: (M4 − M1mg) − (M4 − M1g) = M1g − M1mg, ansambel lima.
  3. Sasaran, prediksi sasaran gabungan: (M4 − M1g) dinilai pada kematian − (M4 − M1g) pada gabungan.
     Sasaran, model kematian dipasang ulang (benih 0–4 saja, pasien saja): (M4 − M1g)_kematian − (M4 − M1g)_gabungan.
  4. AUROC di dalam lipatan (rerata lima lipatan) lawan AUROC gabungan OOF, satu jaringan, benih 0–24: memeriksa
     apakah rekalibrasi per lipatan menurunkan AUROC gabungan M4 lebih dari M1g.
  5. Komponen dinilai lawan kunjungan tanpa kejadian apa pun (komponen lain dikeluarkan), M4 dan M5 lawan M1g,
     25 benih.
  Di awal: jumlah nilai hilang tiap ukuran MUSE (TRIPOD+AI; pohon HistGB menangani NaN sendiri).

Pakai: PYTHONIOENCODING=utf-8 python 14_kontras.py > kontras.log
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
spek = importlib.util.spec_from_file_location('gd', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                 '01_garis_dasar.py'))
gd = importlib.util.module_from_spec(spek)
spek.loader.exec_module(gd)


def ci(v, q=(2.5, 97.5)):
    a, b = np.percentile(v, q)
    return f'[{a:+.3f}; {b:+.3f}]'


def lapor(label, d0, v):
    print(f'  {label}: {d0:+.3f} CI95 {ci(v)} CI90 {ci(v, (5, 95))}', flush=True)


def sampel(g, rng):
    ug, inv = np.unique(g, return_inverse=True)
    urut = np.argsort(inv, kind='stable')
    batas = np.searchsorted(inv[urut], np.arange(len(ug) + 1))
    baris = [urut[batas[j]:batas[j + 1]] for j in range(len(ug))]
    return [np.concatenate([baris[j] for j in rng.integers(0, len(baris), len(baris))]) for _ in range(1000)]


def auc(y, p, i=None):
    return roc_auc_score(y, p) if i is None else roc_auc_score(y[i], p[i])


def main():
    d = gd.kohort()
    k = pd.read_csv(os.path.join(T, 'kohort.csv'))
    assert (d['ed_enc_id'].values == k['ed_enc_id'].values).all()
    y, g = d[S].values, d['patient_ngsci_id'].values
    mati = d['death_030_day'].values
    mt = d['macetrop_030_pos'].astype(bool).astype(int).values
    h = d[gd.NUM].isna()
    print(f'== 0. Nilai hilang ukuran MUSE: {h.sum().to_dict()}; kunjungan dengan >= 1 hilang {h.any(axis=1).sum()}')
    PP = {n: np.load(os.path.join(T, f'oof_{S}_{n}_25.npy')) for n in ('M1g', 'M1mg', 'M4', 'M5')}
    PK = {n: np.load(os.path.join(T, f'oof_death_030_day_{n}.npy')) for n in ('M1g', 'M4')}
    rng = np.random.default_rng(20261102)
    idx = sampel(g, rng)
    pilih = [rng.choice(25, 5, replace=False) for _ in idx]
    satu = rng.integers(0, 25, len(idx))
    E = {n: P[:5].mean(0) for n, P in PP.items()}

    print('== 1. Perataan: (M4 − M1g) ansambel − satu jaringan')
    d_ans = auc(y, E['M4']) - auc(y, E['M1g'])
    d_sat = np.mean([auc(y, PP['M4'][s]) - auc(y, PP['M1g'][s]) for s in range(25)])
    v = [(auc(y, PP['M4'][s].mean(0), i) - auc(y, PP['M1g'][s].mean(0), i))
         - (auc(y, PP['M4'][t], i) - auc(y, PP['M1g'][t], i)) for i, s, t in zip(idx, pilih, satu)]
    lapor('ansambel − satu', d_ans - d_sat, v)

    print('== 2. Pembanding: M1g − M1mg (= (M4 − M1mg) − (M4 − M1g)), ansambel')
    v = [auc(y, PP['M1g'][s].mean(0), i) - auc(y, PP['M1mg'][s].mean(0), i) for i, s in zip(idx, pilih)]
    lapor('M1g − M1mg', auc(y, E['M1g']) - auc(y, E['M1mg']), v)

    print('== 3. Sasaran')
    d0 = (auc(mati, E['M4']) - auc(mati, E['M1g'])) - d_ans
    v = []
    for i, s in zip(idx, pilih):
        a4, a1 = PP['M4'][s].mean(0), PP['M1g'][s].mean(0)
        if mati[i].min() != mati[i].max():
            v.append((auc(mati, a4, i) - auc(mati, a1, i)) - (auc(y, a4, i) - auc(y, a1, i)))
    lapor('prediksi gabungan, kematian − gabungan', d0, v)
    K4, K1 = PK['M4'].mean(0), PK['M1g'].mean(0)
    d0 = (auc(mati, K4) - auc(mati, K1)) - d_ans
    v = [(auc(mati, K4, i) - auc(mati, K1, i)) - (auc(y, E['M4'], i) - auc(y, E['M1g'], i))
         for i in idx if mati[i].min() != mati[i].max()]
    lapor('model kematian dipasang ulang − gabungan (benih 0–4, pasien saja)', d0, v)

    print('== 4. AUROC di dalam lipatan lawan gabungan OOF, satu jaringan, benih 0–24')
    r = {n: [] for n in ('M1g', 'M4')}
    for b in range(25):
        lip = list(StratifiedGroupKFold(5, shuffle=True, random_state=b).split(np.zeros(len(y)), y, g))
        for n in r:
            p = PP[n][b]
            r[n].append((auc(y, p), np.mean([auc(y[te], p[te]) for _, te in lip])))
    for n, v in r.items():
        a = np.array(v)
        print(f'  {n}: gabungan {a[:, 0].mean():.3f}, di dalam lipatan {a[:, 1].mean():.3f}, '
              f'selisih {(a[:, 1] - a[:, 0]).mean():+.3f}')
    a, b = np.array(r['M4']), np.array(r['M1g'])
    print(f'  M4 − M1g satu jaringan: gabungan {(a[:, 0] - b[:, 0]).mean():+.3f}, '
          f'di dalam lipatan {(a[:, 1] - b[:, 1]).mean():+.3f}')

    print('== 5. Komponen lawan kunjungan tanpa kejadian apa pun, ansambel, 25 benih')
    for nama, c, lain in [('macetrop', mt, mati), ('kematian', mati, mt)]:
        m = (c == 1) | (y == 0)
        print(f'  {nama}: {c[m].sum()} kejadian, {(~m).sum()} kunjungan dengan komponen lain saja dikeluarkan')
        ym, gm = c[m], g[m]
        idm = [i for i in sampel(gm, rng)]
        pm = [rng.choice(25, 5, replace=False) for _ in idm]
        print('    AUROC ' + ' '.join(f'{n} {auc(ym, E[n][m]):.3f}' for n in ('M1g', 'M4', 'M5')))
        for a in ('M4', 'M5'):
            v = [auc(ym, PP[a][s][:, m].mean(0), i) - auc(ym, PP['M1g'][s][:, m].mean(0), i)
                 for i, s in zip(idm, pm) if ym[i].min() != ym[i].max()]
            lapor(f'{a} − M1g', auc(ym, E[a][m]) - auc(ym, E['M1g'][m]), v)


if __name__ == '__main__':
    main()
