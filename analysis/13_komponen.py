"""Nightingale ed-bwh-ecg — komponen sasaran gabungan dan angka tambahan atas telaah 31 Okt 2026, tanpa melatih apa pun.

Prediksi model sasaran utama (benih 0–24, 10_benih_tambahan.py; M2 dan M3 benih 0–4). Bootstrap 1.000 ulangan, pasien
sebagai unit, benih bootstrap 20261101; "25 benih" = lima benih berbeda dari 25 tiap ulangan.

  1. Tiap komponen (macetrop_030_pos, death_030_day) dinilai dengan prediksi yang dilatih pada sasaran gabungan,
     semua kunjungan, komponen lain dihitung bukan kejadian: M4 − M1g dan M5 − M1g, 25 benih.
  2. Kunjungan pertama tiap pasien: M4 − M1g dan M5 − M1g, 25 benih.
  3. M2 − M1g dan M3 − M1g, lima benih tetap, pasien saja, CI95 dan CI90.
  4. M4 − M1g pada net benefit per 1.000 kunjungan (ambang 2/5/10/20 %) dan AUPRC, lima benih tetap, pasien saja.

Pakai: venv-gpu python 13_komponen.py > komponen.log
"""
import importlib.util
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

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


def kelompok(g):
    ug, inv = np.unique(g, return_inverse=True)
    urut = np.argsort(inv, kind='stable')
    batas = np.searchsorted(inv[urut], np.arange(len(ug) + 1))
    return [urut[batas[j]:batas[j + 1]] for j in range(len(ug))]


def sampel(g, rng):
    baris = kelompok(g)
    return [np.concatenate([baris[j] for j in rng.integers(0, len(baris), len(baris))]) for _ in range(1000)]


def dua_lima(y, g, PP, rng, pasangan, label):
    idx = [i for i in sampel(g, rng) if y[i].min() != y[i].max()]
    pilih = [rng.choice(25, 5, replace=False) for _ in idx]
    for a, b in pasangan:
        d0 = roc_auc_score(y, PP[a][:5].mean(0)) - roc_auc_score(y, PP[b][:5].mean(0))
        ans = [roc_auc_score(y[i], PP[a][s].mean(0)[i]) - roc_auc_score(y[i], PP[b][s].mean(0)[i])
               for i, s in zip(idx, pilih)]
        print(f'  {label} {a} − {b}: {d0:+.3f} {ci(ans)} CI90 {ci(ans, (5, 95))}', flush=True)


def main():
    d = gd.kohort()
    k = pd.read_csv(os.path.join(T, 'kohort.csv'))
    assert (d['ed_enc_id'].values == k['ed_enc_id'].values).all()
    g = d['patient_ngsci_id'].values
    y = d[S].values
    mt = d['macetrop_030_pos'].astype(bool).astype(int).values
    mati = d['death_030_day'].values
    print(f'gabungan {y.sum()}; macetrop {mt.sum()}; kematian {mati.sum()}; keduanya {(mt & mati).sum()}')
    PP = {n: np.load(os.path.join(T, f'oof_{S}_{n}_25.npy')) for n in ('M1g', 'M4', 'M5')}
    PL = {n: np.load(os.path.join(T, f'oof_{S}_{n}.npy')) for n in ('M1g', 'M2', 'M3', 'M4')}
    pas = [('M4', 'M1g'), ('M5', 'M1g')]
    rng = np.random.default_rng(20261101)

    print('== 1. Komponen, prediksi sasaran gabungan')
    for nama, c in [('macetrop', mt), ('kematian', mati)]:
        print('  ' + nama + ' AUROC ' + ' '.join(f'{n} {roc_auc_score(c, P[:5].mean(0)):.3f}' for n, P in PP.items()))
        dua_lima(c, g, PP, rng, pas, nama)

    print('== 2. Kunjungan pertama')
    u = (pd.DataFrame({'g': g, 't': pd.to_datetime(d['start_datetime'])}).reset_index()
         .sort_values('t').drop_duplicates('g')['index'].sort_values().values)
    dua_lima(y[u], g[u], {n: P[:, u] for n, P in PP.items()}, rng, pas, 'pertama')

    print('== 3. M2, M3 lawan M1g, pasien saja')
    P = {n: v.mean(0) for n, v in PL.items()}
    idx = [i for i in sampel(g, rng) if y[i].min() != y[i].max()]
    for a in ('M2', 'M3'):
        v = [roc_auc_score(y[i], P[a][i]) - roc_auc_score(y[i], P['M1g'][i]) for i in idx]
        print(f'  {a} − M1g {roc_auc_score(y, P[a]) - roc_auc_score(y, P["M1g"]):+.3f} {ci(v)} CI90 {ci(v, (5, 95))}')

    print('== 4. M4 − M1g, net benefit per 1.000 dan AUPRC, pasien saja')
    for pt in [0.02, 0.05, 0.10, 0.20]:
        w = pt / (1 - pt)

        def nb(p, i):
            return ((p[i] >= pt) & (y[i] == 1)).mean() - ((p[i] >= pt) & (y[i] == 0)).mean() * w
        full = np.arange(len(y))
        print(f'  ambang {pt:.2f}: {1000 * (nb(P["M4"], full) - nb(P["M1g"], full)):+.1f} per 1.000 '
              f'{ci([1000 * (nb(P["M4"], i) - nb(P["M1g"], i)) for i in idx])}', flush=True)
    v = [average_precision_score(y[i], P['M4'][i]) - average_precision_score(y[i], P['M1g'][i]) for i in idx]
    print(f'  AUPRC {average_precision_score(y, P["M4"]) - average_precision_score(y, P["M1g"]):+.3f} {ci(v)}')


if __name__ == '__main__':
    main()
