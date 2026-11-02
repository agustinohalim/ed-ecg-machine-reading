"""Nightingale ed-bwh-ecg — kohort dan model garis dasar M0, M1 (rencana: ../Artikel_EDBWH_Rencana.md).

Kohort: kunjungan IGD dengan exclude == False (himpunan analisis pembuat data, 43.451 kunjungan);
satu EKG per kunjungan, yaitu EKG pertama sesudah tiba. Sasaran utama macetrop_pos_or_death_030,
sekunder death_030_day.

  M0   usia + jenis kelamin, regresi logistik
  M1l  M0 + bacaan mesin (sumbu P/R/T, PR, QRS, QT, QTc, laju, 21 pernyataan has_*), logistik
  M1g  ciri yang sama, HistGradientBoosting

StratifiedGroupKFold 5 lipatan (grup = pasien) × 5 benih. Prediksi out-of-fold dirata-ratakan antar
benih; AUROC dan AUPRC juga dilaporkan rata-rata ± SD antar benih. Selang 95 % dari 1.000 bootstrap
pasien. Kalibrasi: Brier dan slope (logit prediksi lawan hasil).

Data tetap di C:/Data/Nightingale (atau NGSCI_DATA); prediksi OOF per kunjungan disimpan di folder
turunan di samping data, bukan di repo. Hanya agregat yang dicetak.
Pakai: python 01_garis_dasar.py > garis_dasar.log
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

D = os.path.join(os.environ.get('NGSCI_DATA', 'C:/Data/Nightingale'), 'ed-bwh-ecg')
R = os.path.join(D, 'v1')
T = os.path.join(D, 'turunan')
SASARAN = ['macetrop_pos_or_death_030', 'death_030_day']
BENIH, LIPAT, BOOT = range(5), 5, 1000
NUM = ['p_axes', 'r_axes', 't_axes', 'pr_interval', 'qrs_duration', 'qt_interval', 'qtc_interval', 'vent_rate']


def kohort():
    e = pd.read_csv(os.path.join(R, 'ed-encounter.csv'))
    e = e[~e['exclude'].astype(bool)]
    m = pd.read_csv(os.path.join(R, 'ecg-metadata.csv'))
    lk = pd.read_csv(os.path.join(R, 'ecg-ed-enc.csv'))
    d = lk.merge(e, on='ed_enc_id').merge(m.drop(columns='patient_ngsci_id'), on='ecg_id')
    d['menit'] = (pd.to_datetime(d['date']) - pd.to_datetime(d['start_datetime'])).dt.total_seconds() / 60
    d = d.sort_values('menit').drop_duplicates('ed_enc_id').sort_values('ed_enc_id').reset_index(drop=True)
    p = pd.read_csv(os.path.join(R, 'patient.csv'))[['patient_ngsci_id', 'sex']]
    d = d.merge(p, on='patient_ngsci_id', how='left')
    d['pria'] = (d['sex'] == 'Male').astype(float)
    idx = pd.read_csv(os.path.join(R, 'ecg-npy-index.csv'))
    d = d.merge(idx, on='ecg_id', how='left')
    for s in SASARAN:
        d[s] = d[s].astype(bool).astype(int)
    return d


def ciri(d):
    has = [c for c in d.columns if c.startswith('has_')]
    return {'M0': ['age_at_admit', 'pria'], 'M1': ['age_at_admit', 'pria'] + NUM + has}


def model(nama):
    if nama == 'M1g':
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
    return make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(),
                         LogisticRegression(max_iter=2000))


def oof(X, y, g, nama):
    P = np.zeros((len(BENIH), len(y)))
    for i, b in enumerate(BENIH):
        for tr, te in StratifiedGroupKFold(LIPAT, shuffle=True, random_state=b).split(X, y, g):
            P[i, te] = model(nama).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return P


def slope(y, p):
    lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return LogisticRegression(C=1e6, max_iter=1000).fit(lg[:, None], y).coef_[0, 0]


def boot(y, p, g, rng):
    """Selang 95 % AUROC/AUPRC, bootstrap pasien."""
    ug, inv = np.unique(g, return_inverse=True)
    baris = [np.flatnonzero(inv == k) for k in range(len(ug))]
    a, b = [], []
    for _ in range(BOOT):
        i = np.concatenate([baris[k] for k in rng.integers(0, len(ug), len(ug))])
        if y[i].min() == y[i].max():
            continue
        a.append(roc_auc_score(y[i], p[i]))
        b.append(average_precision_score(y[i], p[i]))
    return np.percentile(a, [2.5, 97.5]), np.percentile(b, [2.5, 97.5])


def main():
    os.makedirs(T, exist_ok=True)
    d = kohort()
    g = d['patient_ngsci_id'].values
    print(f'kunjungan {len(d)}  pasien {len(np.unique(g))}  EKG tanpa gelombang {d.npy_index.isna().sum()}')
    print(f'menit EKG sesudah tiba: median {d.menit.median():.1f}, P95 {d.menit.quantile(.95):.1f}')
    print(f'usia {d.age_at_admit.mean():.1f} ± {d.age_at_admit.std():.1f}; pria {d.pria.mean():.3f}')
    for s in SASARAN:
        print(f'{s}: {d[s].sum()} kejadian ({d[s].mean():.4f})')
    print(f"diuji (stress/cath ≤ 10 hari): {d['test_010_day'].astype(bool).sum()}")
    d[['ed_enc_id', 'ecg_id', 'patient_ngsci_id', 'npy_index', 'age_at_admit', 'pria', 'test_010_day']
      + SASARAN].to_csv(os.path.join(T, 'kohort.csv'), index=False)
    F = ciri(d)
    rng = np.random.default_rng(20261027)
    for s in SASARAN:
        y = d[s].values
        print(f'\n== {s}')
        print(f"{'model':5} {'AUROC (SD benih)':18} {'CI95':15} {'AUPRC (SD)':16} {'CI95':15} {'Brier':7} slope")
        for nama, kol in [('M0', F['M0']), ('M1l', F['M1']), ('M1g', F['M1'])]:
            X = d[kol].astype(float).values
            P = oof(X, y, g, nama)
            np.save(os.path.join(T, f'oof_{s}_{nama}.npy'), P)
            au = [roc_auc_score(y, p) for p in P]
            ap = [average_precision_score(y, p) for p in P]
            p = P.mean(0)
            ca, cp = boot(y, p, g, rng)
            print(f'{nama:5} {np.mean(au):.3f} ({np.std(au):.3f})    [{ca[0]:.3f}, {ca[1]:.3f}]  '
                  f'{np.mean(ap):.3f} ({np.std(ap):.3f})  [{cp[0]:.3f}, {cp[1]:.3f}]  '
                  f'{brier_score_loss(y, p):.4f}  {slope(y, p):.2f}', flush=True)


if __name__ == '__main__':
    main()
