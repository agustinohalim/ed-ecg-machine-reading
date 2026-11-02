"""Nightingale ed-bwh-ecg — analisis atas telaah ulang 28 Okt 2026 (Telaah/2026-10-28b-EDBWH).

Semua di atas prediksi OOF yang sudah ada (01, 02, 06, 07), sasaran utama. Bootstrap 1.000 ulangan, pasien
sebagai unit, benih bootstrap 20261029.

  1. Selisih AUROC dengan variasi pelatihan: tiap ulangan menarik pasien DAN lima benih dengan pengembalian
     (benih yang sama untuk kedua model, karena lipatannya sama), lalu merata-ratakan prediksi benih
     terpilih. Juga versi satu jaringan: tiap ulangan satu benih acak. Dilaporkan CI95 dan CI90; CI90
     di dalam ±0,02 = setara pada margin 0,02 (logika TOST).
  2. O/E per subkelompok untuk satu jaringan: O/E tiap benih, dilaporkan rerata dan rentang lima benih.
  3. Kontrol penggabungan: M1g + M1l ditumpuk dengan prosedur M5 (dua model bacaan, tanpa gelombang).
  4. Selisih berpasangan M5 − M1g pada kunjungan diuji dan tanpa pasien pencilan; selisih net benefit
     M5 − M1g per 1.000 kunjungan dengan CI95.
  5. Kunjungan pertama tiap pasien saja (menurut waktu kedatangan).
  6. M4 − M1mg (ukuran mesin saja) berpasangan.

Pakai: python 08_telaah2.py > telaah2.log
"""
import importlib.util
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

D = os.path.join(os.environ.get('NGSCI_DATA', 'C:/Data/Nightingale'), 'ed-bwh-ecg')
T = os.path.join(D, 'turunan')
S = 'macetrop_pos_or_death_030'
BOOT, MARGIN = 1000, 0.02
spek = importlib.util.spec_from_file_location('gd', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                 '01_garis_dasar.py'))
gd = importlib.util.module_from_spec(spek)
spek.loader.exec_module(gd)


def kelompok(g):
    ug, inv = np.unique(g, return_inverse=True)
    urut = np.argsort(inv, kind='stable')
    batas = np.searchsorted(inv[urut], np.arange(len(ug) + 1))
    return [urut[batas[k]:batas[k + 1]] for k in range(len(ug))]


def sampel(g, rng, n=BOOT):
    baris = kelompok(g)
    return [np.concatenate([baris[k] for k in rng.integers(0, len(baris), len(baris))]) for _ in range(n)]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ci(v, q=(2.5, 97.5)):
    a, b = np.percentile(v, q)
    return f'[{a:+.3f}; {b:+.3f}]'


def tumpuk(P1, P2, y, g):
    """Sama dengan M5 di 06_cnn12.py: logistik atas dua logit OOF, lipatan baru per benih."""
    out = np.zeros_like(P1)
    for i in range(P1.shape[0]):
        X = np.stack([logit(P1[i]), logit(P2[i])], 1)
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=100 + i).split(X, y, g):
            out[i, te] = LogisticRegression(max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return out


def selisih(y, a, b, idx):
    return [roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]) for i in idx]


def main():
    d = gd.kohort()
    k = pd.read_csv(os.path.join(T, 'kohort.csv'))
    assert (d['ed_enc_id'].values == k['ed_enc_id'].values).all()
    y, g = d[S].values, d['patient_ngsci_id'].values
    PP = {n: np.load(os.path.join(T, f'oof_{S}_{n}.npy')) for n in ['M0', 'M1l', 'M1g', 'M1mg', 'M2', 'M3', 'M4', 'M5']}
    P = {n: v.mean(0) for n, v in PP.items()}
    rng = np.random.default_rng(20261029)
    idx = [i for i in sampel(g, rng) if y[i].min() != y[i].max()]
    nb = PP['M4'].shape[0]

    print('== 1. Selisih AUROC dengan variasi pelatihan (pasien + benih ditarik ulang)')
    pas = [('M4', 'M1g'), ('M5', 'M1g'), ('M2', 'M1g'), ('M4', 'M2'), ('M4', 'M1mg')]
    benih = [rng.integers(0, nb, nb) for _ in idx]
    satu = rng.integers(0, nb, len(idx))
    for a, b in pas:
        ans = [roc_auc_score(y[i], PP[a][s].mean(0)[i]) - roc_auc_score(y[i], PP[b][s].mean(0)[i])
               for i, s in zip(idx, benih)]
        sat = [roc_auc_score(y[i], PP[a][s][i]) - roc_auc_score(y[i], PP[b][s][i]) for i, s in zip(idx, satu)]
        hanya = selisih(y, P[a], P[b], idx)
        d0 = roc_auc_score(y, P[a]) - roc_auc_score(y, P[b])
        d1 = np.mean([roc_auc_score(y, PP[a][s]) - roc_auc_score(y, PP[b][s]) for s in range(nb)])
        lo, hi = np.percentile(ans, [5, 95])
        print(f'  {a} − {b}: ansambel {d0:+.3f}; CI95 pasien saja {ci(hanya)}; pasien + benih {ci(ans)}, '
              f'CI90 {ci(ans, (5, 95))} -> {"di dalam" if -MARGIN < lo and hi < MARGIN else "tidak di dalam"} '
              f'±{MARGIN}; satu jaringan {d1:+.3f} CI95 {ci(sat)} CI90 {ci(sat, (5, 95))}', flush=True)

    print('\n== 2. O/E per subkelompok, satu jaringan (rerata [min; maks] lima benih) lawan ansambel')
    pt = pd.read_csv(os.path.join(D, 'v1', 'patient.csv'))
    e = d[['patient_ngsci_id', 'age_at_admit', 'pria']].merge(pt, on='patient_ngsci_id', how='left',
                                                              validate='many_to_one')
    ras = ['race_black', 'race_hispanic', 'race_white', 'race_other']
    e['ras'] = e[ras].idxmax(axis=1).str.replace('race_', '').where(e[ras].sum(axis=1) == 1)
    e['usia'] = np.where(e.age_at_admit >= 65, '>=65', '<65')
    e['jk'] = np.where(e.pria == 1, 'pria', 'wanita')
    for f in ['jk', 'ras', 'usia']:
        for v in sorted(e[f].dropna().unique()):
            m = (e[f] == v).values
            sel = []
            for n in ('M2', 'M4', 'M5'):
                oe = [y[m].sum() / PP[n][s][m].sum() for s in range(nb)]
                sel.append(f'{n} {np.mean(oe):.2f} [{min(oe):.2f}; {max(oe):.2f}] (ansambel {y[m].sum() / P[n][m].sum():.2f})')
            print(f'  {f:4} {v:9} ' + '  '.join(sel), flush=True)

    print('\n== 3. Kontrol penggabungan: M1g + M1l (dua model bacaan)')
    f = os.path.join(T, f'oof_{S}_Mkontrol.npy')
    PK = np.load(f) if os.path.exists(f) else tumpuk(PP['M1g'], PP['M1l'], y, g)
    np.save(f, PK)
    pk = PK.mean(0)
    print(f'  M1g+M1l AUROC {roc_auc_score(y, pk):.3f}; − M1g {roc_auc_score(y, pk) - roc_auc_score(y, P["M1g"]):+.3f} '
          f'{ci(selisih(y, pk, P["M1g"], idx))}; M5 − (M1g+M1l) '
          f'{roc_auc_score(y, P["M5"]) - roc_auc_score(y, pk):+.3f} {ci(selisih(y, P["M5"], pk, idx))}', flush=True)

    print('\n== 4. M5 − M1g pada analisis kepekaan; net benefit per 1.000 kunjungan')
    u = d['test_010_day'].astype(bool).values
    hit = pd.DataFrame({'g': g, 'y': y}).groupby('g').y.sum()
    top = hit.idxmax()
    for nama, m in [('diuji ≤ 10 hari', u), ('tanpa pasien pencilan', g != top)]:
        im = [i for i in sampel(g[m], rng) if y[m][i].min() != y[m][i].max()]
        for a in ('M4', 'M5'):
            dl = roc_auc_score(y[m], P[a][m]) - roc_auc_score(y[m], P['M1g'][m])
            print(f'  {nama}: {a} − M1g {dl:+.3f} {ci(selisih(y[m], P[a][m], P["M1g"][m], im))}', flush=True)
    for pt_ in [0.02, 0.05, 0.10, 0.20]:
        w = pt_ / (1 - pt_)

        def nbf(p, i):
            return ((p[i] >= pt_) & (y[i] == 1)).mean() - ((p[i] >= pt_) & (y[i] == 0)).mean() * w
        full = np.arange(len(y))
        dl = 1000 * (nbf(P['M5'], full) - nbf(P['M1g'], full))
        bo = [1000 * (nbf(P['M5'], i) - nbf(P['M1g'], i)) for i in idx]
        a, b = np.percentile(bo, [2.5, 97.5])
        print(f'  ambang {pt_:.2f}: M5 − M1g {dl:+.1f} per 1.000 [{a:+.1f}; {b:+.1f}]', flush=True)

    print('\n== 5. Kunjungan pertama tiap pasien')
    pertama = (pd.DataFrame({'g': g, 't': pd.to_datetime(d['start_datetime'])}).reset_index()
               .sort_values('t').drop_duplicates('g')['index'].sort_values().values)
    yf, gf = y[pertama], g[pertama]
    print(f'  n {len(pertama)}, kejadian {yf.sum()}')
    i1 = [i for i in sampel(gf, rng) if yf[i].min() != yf[i].max()]
    for n in ('M0', 'M1g', 'M2', 'M4', 'M5'):
        print(f'  {n} AUROC {roc_auc_score(yf, P[n][pertama]):.3f}')
    for a, b in [('M1g', 'M0'), ('M4', 'M1g'), ('M5', 'M1g')]:
        dl = roc_auc_score(yf, P[a][pertama]) - roc_auc_score(yf, P[b][pertama])
        print(f'  {a} − {b}: {dl:+.3f} {ci(selisih(yf, P[a][pertama], P[b][pertama], i1))}', flush=True)

    print('\n== 6. M4 − M1mg (hanya ukuran mesin)')
    dl = roc_auc_score(y, P['M4']) - roc_auc_score(y, P['M1mg'])
    print(f'  {dl:+.3f} {ci(selisih(y, P["M4"], P["M1mg"], idx))}')


if __name__ == '__main__':
    main()
