"""Nightingale ed-bwh-ecg — analisis tambahan yang diminta telaah 28 Okt 2026 (Telaah/2026-10-28-EDBWH).

Semua di atas prediksi OOF yang sudah ada (01, 02, 06), kecuali M1m yang dilatih di sini.

  1. M1m: usia + jenis kelamin + 8 ukuran mesin saja (tanpa 21 bendera has_*), logistik dan
     gradient boosting, lipatan dan benih sama dengan 01. Bendera has_* adalah kata kunci di
     "cardiology remarks" yang asal-usulnya (pernyataan komputer atau bacaan kardiolog) tidak
     dinyatakan dokumentasi dataset; ukuran interval/sumbu adalah keluaran mesin.
  2. Satu jaringan lawan ansambel: AUROC, slope, dan ΔAUROC terhadap M1g untuk tiap benih M2
     (dan M4 bila ada), CI95 bootstrap pasien per benih.
  3. Kalibrasi: intercept (calibration-in-the-large, logit offset) dan slope tiap model.
  4. O/E per subkelompok dengan CI95 bootstrap pasien (subkelompok sama dengan 03_subkelompok.py).
  5. Hanya kunjungan diuji ≤ 10 hari: AUROC dengan CI95 dan ΔAUROC.
  6. Tanpa pasien dengan kejadian terbanyak (86 kunjungan, 77 kejadian): AUROC dengan CI95.
  7. Net benefit (decision curve) pada ambang 2, 5, 10, 20 %.

Sasaran utama kecuali M1m (keduanya). Bootstrap 1.000 ulangan, pasien sebagai unit.
Pakai: python 07_tambahan.py > tambahan.log
"""
import importlib.util
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

D = os.path.join(os.environ.get('NGSCI_DATA', 'C:/Data/Nightingale'), 'ed-bwh-ecg')
T = os.path.join(D, 'turunan')
S = 'macetrop_pos_or_death_030'
BOOT = 1000
spek = importlib.util.spec_from_file_location('gd', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                 '01_garis_dasar.py'))
gd = importlib.util.module_from_spec(spek)
spek.loader.exec_module(gd)


def kelompok(g):
    ug, inv = np.unique(g, return_inverse=True)
    urut = np.argsort(inv, kind='stable')
    batas = np.searchsorted(inv[urut], np.arange(len(ug) + 1))
    return [urut[batas[k]:batas[k + 1]] for k in range(len(ug))]


def sampel(g, rng):
    baris = kelompok(g)
    return [np.concatenate([baris[k] for k in rng.integers(0, len(baris), len(baris))]) for _ in range(BOOT)]


def ci(v):
    a, b = np.percentile(v, [2.5, 97.5])
    return f'[{a:.3f}; {b:.3f}]'


def ci_d(v):
    a, b = np.percentile(v, [2.5, 97.5])
    return f'[{a:+.3f}; {b:+.3f}]'


def auroc_ci(y, cols, g, rng, pasangan=()):
    bs = [i for i in sampel(g, rng) if y[i].min() != y[i].max()]
    au = {n: [roc_auc_score(y[i], p[i]) for i in bs] for n, p in cols.items()}
    for n, p in cols.items():
        print(f'  {n:8} AUROC {roc_auc_score(y, p):.3f} {ci(au[n])}')
    for a, b in pasangan:
        dlt = roc_auc_score(y, cols[a]) - roc_auc_score(y, cols[b])
        print(f'  {a} − {b}: {dlt:+.3f} {ci_d(np.subtract(au[a], au[b]))}')


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def kalibrasi(y, p):
    lg = logit(p)
    sl = LogisticRegression(C=1e6, max_iter=1000).fit(lg[:, None], y).coef_[0, 0]
    # intercept dengan logit prediksi sebagai offset: cari a sehingga sum sigmoid(lg + a) = sum y
    a = 0.0
    for _ in range(50):
        q = 1 / (1 + np.exp(-(lg + a)))
        a -= (q.sum() - y.sum()) / (q * (1 - q)).sum()
    return a, sl


def main():
    d = gd.kohort()
    k = pd.read_csv(os.path.join(T, 'kohort.csv'))
    assert (d['ed_enc_id'].values == k['ed_enc_id'].values).all()
    g = d['patient_ngsci_id'].values
    rng = np.random.default_rng(20261028)

    print('== 1. M1m: usia + jenis kelamin + 8 ukuran mesin (tanpa bendera has_*)')
    X = d[['age_at_admit', 'pria'] + gd.NUM].astype(float).values
    for s in gd.SASARAN:
        y = d[s].values
        cols = {}
        for nama, inti in [('M1ml', 'M1l'), ('M1mg', 'M1g')]:
            f = os.path.join(T, f'oof_{s}_{nama}.npy')
            P = np.load(f) if os.path.exists(f) else gd.oof(X, y, g, inti)
            np.save(f, P)
            print(f'{s} {nama}: AUROC per benih {np.mean([roc_auc_score(y, p) for p in P]):.3f}')
            cols[nama] = P.mean(0)
        for n in ('M0', 'M1g'):
            cols[n] = np.load(os.path.join(T, f'oof_{s}_{n}.npy')).mean(0)
        print(s)
        auroc_ci(y, cols, g, rng, [('M1mg', 'M0'), ('M1g', 'M1mg')])

    y = d[S].values
    nama = [n for n in ['M0', 'M1g', 'M1mg', 'M2', 'M3', 'M4', 'M5']
            if os.path.exists(os.path.join(T, f'oof_{S}_{n}.npy'))]
    PP = {n: np.load(os.path.join(T, f'oof_{S}_{n}.npy')) for n in nama}
    P = {n: v.mean(0) for n, v in PP.items()}

    print(f'\n== 2. Satu jaringan lawan ansambel ({S})')
    bs = [i for i in sampel(g, rng) if y[i].min() != y[i].max()]
    for n in [m for m in ('M2', 'M4') if m in PP]:
        for b in range(PP[n].shape[0]):
            p, q = PP[n][b], PP['M1g'][b]
            dl = [roc_auc_score(y[i], p[i]) - roc_auc_score(y[i], q[i]) for i in bs[:200]]
            a, sl = kalibrasi(y, p)
            print(f'  {n} benih {b}: AUROC {roc_auc_score(y, p):.3f}; − M1g benih {b} '
                  f'{roc_auc_score(y, p) - roc_auc_score(y, q):+.3f} {ci_d(dl)} (200 ulangan); '
                  f'intercept {a:+.2f} slope {sl:.2f}')

    print('\n== 3. Kalibrasi ansambel: intercept (offset logit; 0 = rata-rata tepat) dan slope')
    for n in nama:
        a, sl = kalibrasi(y, P[n])
        print(f'  {n:5} intercept {a:+.3f}  slope {sl:.2f}  rerata prediksi {P[n].mean():.4f} '
              f'(teramati {y.mean():.4f})')

    print('\n== 4. O/E per subkelompok, CI95 bootstrap pasien')
    pt = pd.read_csv(os.path.join(D, 'v1', 'patient.csv'))
    e = d[['patient_ngsci_id', 'age_at_admit', 'pria']].merge(pt, on='patient_ngsci_id', how='left',
                                                              validate='many_to_one')
    ras = ['race_black', 'race_hispanic', 'race_white', 'race_other']
    e['ras'] = e[ras].idxmax(axis=1).str.replace('race_', '').where(e[ras].sum(axis=1) == 1)
    e['usia'] = np.where(e.age_at_admit >= 65, '>=65', '<65')
    e['jk'] = np.where(e.pria == 1, 'pria', 'wanita')
    batas = e.drop_duplicates('patient_ngsci_id').agi_under_25k.quantile([1 / 3, 2 / 3]).values
    e['pendapatan'] = pd.cut(e.agi_under_25k, [-np.inf, *batas, np.inf],
                             labels=['tinggi', 'menengah', 'rendah']).astype(object)
    oe_m = [n for n in ('M1g', 'M2', 'M3', 'M4', 'M5') if n in P]
    print(f"  {'faktor':10} {'nilai':9} " + '  '.join(f'{n:>20}' for n in oe_m))
    for f in ['jk', 'ras', 'usia', 'pendapatan']:
        for v in sorted(e[f].dropna().unique()):
            m = (e[f] == v).values
            ym, gm = y[m], g[m]
            idx = sampel(gm, rng)
            sel = []
            for n in oe_m:
                pm = P[n][m]
                bo = [ym[i].sum() / pm[i].sum() for i in idx]
                a, b = np.percentile(bo, [2.5, 97.5])
                sel.append(f'{ym.sum() / pm.sum():.2f} [{a:.2f}; {b:.2f}]')
            print(f'  {f:10} {v:9} ' + '  '.join(f'{x:>20}' for x in sel), flush=True)

    print('\n== 5. Hanya kunjungan diuji ≤ 10 hari')
    u = d['test_010_day'].astype(bool).values
    print(f'  n {u.sum()}, kejadian {y[u].sum()}')
    pas = [('M1g', 'M0'), ('M2', 'M1g'), ('M3', 'M1g')] + ([('M4', 'M1g')] if 'M4' in P else [])
    auroc_ci(y[u], {n: P[n][u] for n in nama}, g[u], rng, pas)

    print('\n== 6. Tanpa pasien dengan kejadian terbanyak')
    hit = pd.DataFrame({'g': g, 'y': y}).groupby('g').y.agg(['size', 'sum']).sort_values('sum')
    top = hit.index[-1]
    print(f'  dikeluarkan: 1 pasien, {hit.loc[top, "size"]} kunjungan, {hit.loc[top, "sum"]} kejadian; '
          f'berikutnya {list(hit["sum"].values[-4:-1][::-1])} kejadian')
    k_ = g != top
    auroc_ci(y[k_], {n: P[n][k_] for n in nama}, g[k_], rng, pas)

    print('\n== 7. Net benefit = TP/n − FP/n · pt/(1−pt); bandingkan dengan "uji semua"')
    print(f"  {'ambang':6} {'semua':>7} " + ' '.join(f'{n:>7}' for n in nama))
    for pt_ in [0.02, 0.05, 0.10, 0.20]:
        w = pt_ / (1 - pt_)
        semua = y.mean() - (1 - y.mean()) * w
        nb = [((P[n] >= pt_) & (y == 1)).mean() - ((P[n] >= pt_) & (y == 0)).mean() * w for n in nama]
        print(f'  {pt_:6.2f} {semua:7.4f} ' + ' '.join(f'{v:7.4f}' for v in nb))


if __name__ == '__main__':
    main()
