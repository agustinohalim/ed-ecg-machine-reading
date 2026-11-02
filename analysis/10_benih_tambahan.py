"""Nightingale ed-bwh-ecg — bootstrap variasi pelatihan dengan ukuran ansambel tetap lima.

Menjawab telaah 30 Okt 2026 (Telaah/2026-10-30-EDBWH-Kinetik): 08_telaah2.py menarik lima benih dari lima
DENGAN pengembalian, sehingga tiap ulangan rata-rata hanya memuat ±3,4 jaringan berbeda dan selang "pasien +
benih" tergeser ke arah kinerja satu jaringan. Di sini kumpulan benih diperbesar menjadi 25 dan tiap ulangan
menarik lima benih BERBEDA tanpa pengembalian, jadi tiap ulangan adalah ansambel lima jaringan sungguhan.

  - M4 (06_cnn12.py) dilatih untuk benih 5–24 di GPU; benih 0–4 dari berkas yang sudah ada. Tiap benih dan
    lipatan disimpan begitu selesai (oof_<s>_M4_benih<b>[_lipat<k>].npy), jadi proses bisa dilanjutkan.
  - M1g dan M1mg (CPU, menit) dipasang ulang pada lipatan benih 5–24 yang sama, supaya selisih tetap
    berpasangan per benih; M5 ditumpuk per benih seperti 06_cnn12.py (random_state 100 + benih).
  - M2 tidak dilatih ulang (±9 jam lagi): pasangan dengan M2 tetap memakai selang pasien saja.
  - Bootstrap 1.000 ulangan, pasien sebagai unit, benih bootstrap 20261031. Dilaporkan CI95 dan CI90;
    CI90 di dalam ±0,02 = setara pada margin 0,02 (logika TOST, margin pasca-hoc).

Sasaran utama saja. Pakai: venv-gpu python 10_benih_tambahan.py > benih_tambahan.log
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
LAMA, BARU = range(5), range(5, 25)
BOOT, ANS, MARGIN = 1000, 5, 0.02


def muat(nama):
    spek = importlib.util.spec_from_file_location(nama, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                      f'{nama}.py'))
    m = importlib.util.module_from_spec(spek)
    spek.loader.exec_module(m)
    return m


gd, c12 = muat('01_garis_dasar'), muat('06_cnn12')


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ci(v, q=(2.5, 97.5)):
    a, b = np.percentile(v, q)
    return f'[{a:+.3f}; {b:+.3f}]'


def gabung(nama, baru):
    """25 × n: benih 0–4 dari berkas lama, 5–24 dari baru; disimpan dengan akhiran _25."""
    P = np.concatenate([np.load(os.path.join(T, f'oof_{S}_{nama}.npy')), baru])
    np.save(os.path.join(T, f'oof_{S}_{nama}_25.npy'), P)
    return P


def main():
    d = gd.kohort()
    k = pd.read_csv(os.path.join(T, 'kohort.csv'))
    assert (d['ed_enc_id'].values == k['ed_enc_id'].values).all()
    y, g = d[S].values, d['patient_ngsci_id'].values
    print(f'{S}: {len(y)} kunjungan, {y.sum()} kejadian; benih baru {BARU.start}–{BARU.stop - 1}; '
          f'alat {c12.ALAT}', flush=True)

    gd.BENIH = BARU
    F = gd.ciri(d)
    P1 = gabung('M1g', gd.oof(d[F['M1']].astype(float).values, y, g, 'M1g'))
    P1m = gabung('M1mg', gd.oof(d[['age_at_admit', 'pria'] + gd.NUM].astype(float).values, y, g, 'M1g'))
    print('M1g, M1mg benih baru selesai', flush=True)

    c12.BENIH = BARU
    W = c12.siapkan(k)
    Z = np.stack([k['age_at_admit'].values / 100, k['pria'].values], 1).astype(np.float32)
    P4 = gabung('M4', c12.oof_cnn(W, Z, y, g, S))

    P5b = np.zeros((len(BARU), len(y)))
    for i, b in enumerate(BARU):
        X = np.stack([logit(P1[b]), logit(P4[b])], 1)
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=100 + b).split(X, y, g):
            P5b[i, te] = LogisticRegression(max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    P5 = gabung('M5', P5b)
    PP = {'M1g': P1, 'M1mg': P1m, 'M4': P4, 'M5': P5}
    nb = len(LAMA) + len(BARU)

    print('\nAUROC per benih (rerata ± SD, 25 benih); benih 0–4 lawan 5–24')
    for n, P in PP.items():
        au = np.array([roc_auc_score(y, p) for p in P])
        print(f'  {n:5} {au.mean():.3f} ± {au.std():.3f}; 0–4 {au[:5].mean():.3f}, 5–24 {au[5:].mean():.3f}')

    rng = np.random.default_rng(20261031)
    baris = c12.kelompok(g)
    idx = [np.concatenate([baris[j] for j in rng.integers(0, len(baris), len(baris))]) for _ in range(BOOT)]
    idx = [i for i in idx if y[i].min() != y[i].max()]
    pilih = [rng.choice(nb, ANS, replace=False) for _ in idx]
    satu = rng.integers(0, nb, len(idx))

    print(f'\nSelisih AUROC, {len(idx)} ulangan; ansambel = {ANS} benih berbeda dari {nb}')
    for a, b in [('M4', 'M1g'), ('M5', 'M1g'), ('M4', 'M1mg')]:
        asli = roc_auc_score(y, PP[a][:5].mean(0)) - roc_auc_score(y, PP[b][:5].mean(0))
        hanya = [roc_auc_score(y[i], PP[a][:5].mean(0)[i]) - roc_auc_score(y[i], PP[b][:5].mean(0)[i]) for i in idx]
        # nilai harapan ansambel lima: rerata selisih atas 200 himpunan lima benih acak, data penuh
        harap = np.mean([roc_auc_score(y, PP[a][s].mean(0)) - roc_auc_score(y, PP[b][s].mean(0))
                         for s in (rng.choice(nb, ANS, replace=False) for _ in range(200))])
        ans = [roc_auc_score(y[i], PP[a][s].mean(0)[i]) - roc_auc_score(y[i], PP[b][s].mean(0)[i])
               for i, s in zip(idx, pilih)]
        sat = [roc_auc_score(y[i], PP[a][s][i]) - roc_auc_score(y[i], PP[b][s][i]) for i, s in zip(idx, satu)]
        d1 = np.mean([roc_auc_score(y, PP[a][s]) - roc_auc_score(y, PP[b][s]) for s in range(nb)])
        lo, hi = np.percentile(ans, [5, 95])
        print(f'  {a} − {b}: ansambel benih 0–4 {asli:+.3f}, pasien saja {ci(hanya)}; harapan ansambel lima '
              f'{harap:+.3f}, pasien + benih {ci(ans)} CI90 {ci(ans, (5, 95))} -> '
              f'{"di dalam" if -MARGIN < lo and hi < MARGIN else "tidak di dalam"} ±{MARGIN}; '
              f'satu jaringan {d1:+.3f} {ci(sat)} CI90 {ci(sat, (5, 95))}', flush=True)


if __name__ == '__main__':
    main()
