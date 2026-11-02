"""Nightingale ed-bwh-ecg — kinerja M0/M1g/M2/M3/M4/M5 per subkelompok (rencana: ../Artikel_EDBWH_Rencana.md, A).

Subkelompok ditetapkan 28 Okt 2026 sebelum dijalankan: jenis kelamin; ras (kolom race_* patient.csv,
tepat satu bernilai 1); usia < 65 / ≥ 65; pendapatan wilayah = tertil porsi AGI < USD 25 rb di blok sensus alamat
pasien (agi_under_25k; porsi tinggi = wilayah berpendapatan rendah; pasien tanpa nilai dikeluarkan dari
analisis pendapatan saja). Sasaran utama saja. Prediksi = OOF dirata-ratakan atas 5 benih, dari
01_garis_dasar.py dan 02_cnn.py; tidak ada model yang dilatih ulang.

Per subkelompok: n, kejadian, AUROC tiap model dengan CI95 bootstrap pasien (1.000 ulangan, pasien
diambil ulang di dalam subkelompok), dan O/E = kejadian teramati / jumlah probabilitas terduga untuk
M1g dan M3 (1 = terkalibrasi rata-rata; > 1 = model meremehkan risiko subkelompok itu).

Pakai: python 03_subkelompok.py > subkelompok.log
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

D = os.path.join(os.environ.get('NGSCI_DATA', 'C:/Data/Nightingale'), 'ed-bwh-ecg')
T = os.path.join(D, 'turunan')
S = 'macetrop_pos_or_death_030'
MODEL, BOOT = ['M0', 'M1g', 'M2', 'M3', 'M4', 'M5'], 1000  # M4/M5 ditambah 28 Okt; indeks bootstrap tidak bergantung jumlah model


def kelompok(g):
    ug, inv = np.unique(g, return_inverse=True)
    urut = np.argsort(inv, kind='stable')
    batas = np.searchsorted(inv[urut], np.arange(len(ug) + 1))
    return [urut[batas[k]:batas[k + 1]] for k in range(len(ug))]


def ci_auroc(y, cols, g, rng):
    baris = kelompok(g)
    hasil = {n: [] for n in cols}
    for _ in range(BOOT):
        i = np.concatenate([baris[k] for k in rng.integers(0, len(baris), len(baris))])
        if y[i].min() == y[i].max():
            continue
        for n, p in cols.items():
            hasil[n].append(roc_auc_score(y[i], p[i]))
    return {n: np.percentile(v, [2.5, 97.5]) for n, v in hasil.items()}


def main():
    d = pd.read_csv(os.path.join(T, 'kohort.csv'))
    p = pd.read_csv(os.path.join(D, 'v1', 'patient.csv'))
    d = d.merge(p, on='patient_ngsci_id', how='left', validate='many_to_one')
    ras = ['race_black', 'race_hispanic', 'race_white', 'race_other']
    d['ras'] = d[ras].idxmax(axis=1).str.replace('race_', '').where(d[ras].sum(axis=1) == 1)
    d['usia'] = np.where(d.age_at_admit >= 65, '>=65', '<65')
    d['jk'] = np.where(d.pria == 1, 'pria', 'wanita')
    batas = d.drop_duplicates('patient_ngsci_id').agi_under_25k.quantile([1 / 3, 2 / 3]).values
    d['pendapatan'] = pd.cut(d.agi_under_25k, [-np.inf, *batas, np.inf],
                             labels=['tinggi', 'menengah', 'rendah']).astype(object)
    print(f'{S}: {len(d)} kunjungan; batas tertil agi_under_25k {batas[0]:.3f} / {batas[1]:.3f} '
          f'(tertil atas = wilayah berpendapatan rendah); tanpa ras {d.ras.isna().sum()}, '
          f'tanpa pendapatan {d.pendapatan.isna().sum()}')
    y, g = d[S].values, d['patient_ngsci_id'].values
    P = {n: np.load(os.path.join(T, f'oof_{S}_{n}.npy')).mean(0) for n in MODEL}
    rng = np.random.default_rng(20261028)
    print(f"\n{'faktor':10} {'nilai':9} {'n':>6} {'kejadian':>9}  " + '  '.join(f'{n:>21}' for n in MODEL)
          + '   O/E M1g  O/E M3')
    for f in ['jk', 'ras', 'usia', 'pendapatan']:
        for v in sorted(d[f].dropna().unique()):
            m = (d[f] == v).values
            cols = {n: P[n][m] for n in MODEL}
            ci = ci_auroc(y[m], cols, g[m], rng)
            sel = '  '.join(f'{roc_auc_score(y[m], cols[n]):.3f} [{ci[n][0]:.3f}; {ci[n][1]:.3f}]'
                            for n in MODEL)
            oe = [y[m].sum() / P[n][m].sum() for n in ('M1g', 'M3')]
            print(f'{f:10} {v:9} {m.sum():6d} {y[m].sum():5d} ({y[m].mean():.1%})  {sel}   '
                  f'{oe[0]:.2f}     {oe[1]:.2f}', flush=True)


if __name__ == '__main__':
    main()
