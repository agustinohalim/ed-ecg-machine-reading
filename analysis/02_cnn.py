"""Nightingale ed-bwh-ecg — M2 (CNN 1D gelombang + usia + jenis kelamin) dan M3 (gabungan M1g + M2).

Kohort dan prediksi M0/M1 dari 01_garis_dasar.py (folder turunan). Gelombang V1, II, V5, 10 detik
100 Hz (3 × 1.000), dikurangi rerata per sadapan, dibagi simpangan baku global data latih lipatan.
Arsitektur seperti Uji_Arrest/02_cnn_arrest.py: empat blok Conv1d(kernel 7) + BatchNorm + ReLU +
MaxPool(4), kanal 32-64-64-128, rerata global; usia/100 dan jenis kelamin disambung sebelum lapis
linear, dropout 0,3. Adam 1e-3, EPOCH epoch, batch 128, pos_weight seimbang. Lipatan sama dengan
01 (StratifiedGroupKFold 5, grup = pasien, benih 0–4). Berjalan di CPU.

M3: regresi logistik atas logit OOF M1g dan M2, dinilai dengan lipatan pasien kedua (benih 100+).
ΔAUROC berpasangan (M1g−M0, M2−M0, M3−M1g) dengan 1.000 bootstrap pasien.

Pakai: python 02_cnn.py [sasaran] > cnn.log   (sasaran baku macetrop_pos_or_death_030)
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

D = os.path.join(os.environ.get('NGSCI_DATA', 'C:/Data/Nightingale'), 'ed-bwh-ecg')
T = os.path.join(D, 'turunan')
BENIH, LIPAT, BOOT = range(5), 5, 1000
EPOCH, BATCH = 15, 128
torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))


class Jaring(nn.Module):
    def __init__(self):
        super().__init__()
        lap, c0 = [], 3
        for c in [32, 64, 64, 128]:
            lap += [nn.Conv1d(c0, c, 7, padding=3), nn.BatchNorm1d(c), nn.ReLU(), nn.MaxPool1d(4)]
            c0 = c
        self.fitur = nn.Sequential(*lap)
        self.kepala = nn.Sequential(nn.Dropout(0.3), nn.Linear(128 + 2, 1))

    def forward(self, x, z):
        return self.kepala(torch.cat([self.fitur(x).mean(-1), z], 1)).squeeze(-1)


def latih_prediksi(xa, za, y, fit, uji, benih):
    """xa, za: tensor seluruh kohort (tidak disalin per lipatan); fit, uji: indeks baris."""
    torch.manual_seed(benih)
    sd = float(xa[torch.from_numpy(fit[::10])].std())
    ytr = y[fit]
    m = Jaring()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    pos = ytr.mean()
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos, dtype=torch.float32))
    ft, yt = torch.from_numpy(fit), torch.tensor(ytr, dtype=torch.float32)
    for _ in range(EPOCH):
        m.train()
        perm = torch.randperm(len(yt))
        for i in range(0, len(yt), BATCH):
            j = perm[i:i + BATCH]
            r = ft[j]
            opt.zero_grad()
            loss(m(xa[r] / sd, za[r]), yt[j]).backward()
            opt.step()
    m.eval()
    out = []
    with torch.no_grad():
        ut = torch.from_numpy(uji)
        for i in range(0, len(ut), 1024):
            r = ut[i:i + 1024]
            out.append(m(xa[r] / sd, za[r]).numpy())
    return np.concatenate(out)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def oof_cnn(W, Z, y, g, s):
    """Tiap benih disimpan begitu selesai, supaya proses yang terhenti dapat dilanjutkan."""
    xa, za = torch.from_numpy(W), torch.from_numpy(Z)
    P = np.zeros((len(BENIH), len(y)))
    for i, b in enumerate(BENIH):
        f = os.path.join(T, f'oof_{s}_M2_benih{b}.npy')
        if os.path.exists(f):
            P[i] = np.load(f)
            print(f'  benih {b}: dari berkas', flush=True)
            continue
        for k, (tr, te) in enumerate(StratifiedGroupKFold(LIPAT, shuffle=True, random_state=b).split(W, y, g)):
            t0 = time.time()
            # CNN berbobot kelas memberi logit yang tidak terkalibrasi; 10 % pasien latih disisihkan
            # untuk memetakan logit ke probabilitas dengan regresi logistik
            rng = np.random.default_rng(1000 * b + k)
            ug = np.unique(g[tr])
            cal_g = set(rng.choice(ug, len(ug) // 10, replace=False))
            cal = np.array([x in cal_g for x in g[tr]])
            fit, cl = tr[~cal], tr[cal]
            out = latih_prediksi(xa, za, y, fit, np.concatenate([cl, te]), b)
            lr = LogisticRegression(C=1e6, max_iter=1000).fit(out[:len(cl), None], y[cl])
            P[i, te] = lr.predict_proba(out[len(cl):, None])[:, 1]
            print(f'  benih {b} lipat {k}: AUROC {roc_auc_score(y[te], P[i, te]):.3f}  '
                  f'({time.time() - t0:.0f} d, latih {len(fit)} kalibrasi {len(cl)})', flush=True)
        np.save(f, P[i])
    return P


def kelompok(g):
    ug, inv = np.unique(g, return_inverse=True)
    urut = np.argsort(inv, kind='stable')
    batas = np.searchsorted(inv[urut], np.arange(len(ug) + 1))
    return [urut[batas[k]:batas[k + 1]] for k in range(len(ug))]


def boot(y, cols, g, rng):
    """Bootstrap pasien: AUROC/AUPRC tiap kolom dan selisih AUROC antar pasangan."""
    baris = kelompok(g)
    hasil = []
    for _ in range(BOOT):
        i = np.concatenate([baris[k] for k in rng.integers(0, len(baris), len(baris))])
        if y[i].min() == y[i].max():
            continue
        hasil.append({n: (roc_auc_score(y[i], p[i]), average_precision_score(y[i], p[i])) for n, p in cols.items()})
    return hasil


def slope(y, p):
    return LogisticRegression(C=1e6, max_iter=1000).fit(logit(p)[:, None], y).coef_[0, 0]


def main():
    s = sys.argv[1] if len(sys.argv) > 1 else 'macetrop_pos_or_death_030'
    d = pd.read_csv(os.path.join(T, 'kohort.csv'))
    y, g = d[s].values, d['patient_ngsci_id'].values
    w = np.load(os.path.join(D, 'v1', 'ecg-waveform.npy'), mmap_mode='r')
    W = np.asarray(w[d['npy_index'].astype(int).values], dtype=np.float32)
    del w
    W -= W.mean(-1, keepdims=True)
    W = np.nan_to_num(W)
    Z = np.stack([d['age_at_admit'].values / 100, d['pria'].values], 1).astype(np.float32)
    print(f'{s}: {len(y)} kunjungan, {y.sum()} kejadian; gelombang {W.shape}', flush=True)
    f = os.path.join(T, f'oof_{s}_M2.npy')
    if os.path.exists(f):
        P2 = np.load(f)
    else:
        P2 = oof_cnn(W, Z, y, g, s)
        np.save(f, P2)
    P0 = np.load(os.path.join(T, f'oof_{s}_M0.npy'))
    P1 = np.load(os.path.join(T, f'oof_{s}_M1g.npy'))
    # M3: logistik atas logit OOF M1g dan M2, lipatan pasien kedua
    P3 = np.zeros_like(P2)
    for i, b in enumerate(BENIH):
        X = np.stack([logit(P1[i]), logit(P2[i])], 1)
        for tr, te in StratifiedGroupKFold(LIPAT, shuffle=True, random_state=100 + b).split(X, y, g):
            P3[i, te] = LogisticRegression(max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    np.save(os.path.join(T, f'oof_{s}_M3.npy'), P3)
    semua = {'M0': P0, 'M1g': P1, 'M2': P2, 'M3': P3}
    rata = {n: P.mean(0) for n, P in semua.items()}
    bs = boot(y, rata, g, np.random.default_rng(20261027))
    print(f"\n{'model':5} {'AUROC (SD benih)':18} {'CI95':15} {'AUPRC (SD)':16} {'CI95':15} {'Brier':7} slope")
    for n, P in semua.items():
        au = [roc_auc_score(y, p) for p in P]
        ap = [average_precision_score(y, p) for p in P]
        ca = np.percentile([h[n][0] for h in bs], [2.5, 97.5])
        cp = np.percentile([h[n][1] for h in bs], [2.5, 97.5])
        print(f'{n:5} {np.mean(au):.3f} ({np.std(au):.3f})    [{ca[0]:.3f}, {ca[1]:.3f}]  '
              f'{np.mean(ap):.3f} ({np.std(ap):.3f})  [{cp[0]:.3f}, {cp[1]:.3f}]  '
              f'{brier_score_loss(y, rata[n]):.4f}  {slope(y, rata[n]):.2f}')
    print('\nΔAUROC berpasangan (prediksi rata-rata benih), CI95 bootstrap pasien')
    for a, b in [('M1g', 'M0'), ('M2', 'M0'), ('M2', 'M1g'), ('M3', 'M1g')]:
        dlt = roc_auc_score(y, rata[a]) - roc_auc_score(y, rata[b])
        ci = np.percentile([h[a][0] - h[b][0] for h in bs], [2.5, 97.5])
        print(f'{a} − {b}: {dlt:+.3f}  [{ci[0]:+.3f}, {ci[1]:+.3f}]')
    # uji kepekaan: hanya kunjungan yang diuji stress/cath ≤ 10 hari
    u = d['test_010_day'].astype(bool).values
    print(f'\nKepekaan — hanya kunjungan diuji (n {u.sum()}, kejadian {y[u].sum()}):')
    for n in semua:
        print(f'  {n}: AUROC {roc_auc_score(y[u], rata[n][u]):.3f}')


if __name__ == '__main__':
    main()
