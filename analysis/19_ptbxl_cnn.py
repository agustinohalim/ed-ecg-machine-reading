"""PTB-XL+ — CNN-12 (arsitektur M4) per putaran, untuk replikasi kurva ukuran ansambel (prapendaftaran 2 Nov 2026).

Masukan dari 18_ptbxl_siapkan.py: turunan/gelombang12.npy (N × 12 × 5.000, float16, rerata tiap sadapan dikurangkan)
dan turunan/kohort.csv. Arsitektur dan pelatihan = 06_cnn12.py (M4): lima blok Conv1d(kernel 7) + BatchNorm + ReLU +
MaxPool(4), kanal 32-64-64-128-128, rerata global, usia/100 dan jenis kelamin disambung, dropout 0,3; Adam 1e-3,
batch 128, 15 epoch, pos_weight; kalibrasi ulang logistik pada 10 % pasien lipatan latih. Satu-satunya perbedaan:
12 kanal masukan (sadapan penuh 10 detik), bukan 15 baris susunan kertas.

Putaran b = StratifiedGroupKFold(5, shuffle, random_state=b), grup = pasien, + torch.manual_seed(b). Tiap lipatan
disimpan begitu selesai (oof_<sasaran>_cnn_benih<b>_lipat<k>.npy), jadi proses bisa dihentikan dan dilanjutkan.

Pakai: venv-gpu python 19_ptbxl_cnn.py <sasaran> <jumlah putaran> > ptbxl_cnn_<sasaran>.log
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

T = os.path.join(os.environ.get('PTBXL_DATA', 'C:/Data/PhysioNet/ptbxl'), 'turunan')
LIPAT, EPOCH, BATCH = 5, 15, 128
ALAT = 'cuda' if torch.cuda.is_available() else 'cpu'


class Jaring(nn.Module):
    def __init__(self, masuk=12):
        super().__init__()
        lap, c0 = [], masuk
        for c in [32, 64, 64, 128, 128]:
            lap += [nn.Conv1d(c0, c, 7, padding=3), nn.BatchNorm1d(c), nn.ReLU(), nn.MaxPool1d(4)]
            c0 = c
        self.fitur = nn.Sequential(*lap)
        self.kepala = nn.Sequential(nn.Dropout(0.3), nn.Linear(128 + 2, 1))

    def forward(self, x, z):
        return self.kepala(torch.cat([self.fitur(x).mean(-1), z], 1)).squeeze(-1)


def ambil(W, Z, r, sd):
    r = np.sort(r)
    x = torch.from_numpy(np.asarray(W[r], dtype=np.float32)).to(ALAT) / sd
    return x, torch.from_numpy(Z[r]).to(ALAT), r


def latih_prediksi(W, Z, y, fit, uji, benih):
    torch.manual_seed(benih)
    rng = np.random.default_rng(benih)
    sub = np.sort(rng.choice(fit, min(2000, len(fit)), replace=False))
    x = np.asarray(W[sub], dtype=np.float32)
    sd = float(x[x != 0].std())
    m = Jaring().to(ALAT)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    pos = y[fit].mean()
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos, dtype=torch.float32, device=ALAT))
    yt = torch.tensor(y, dtype=torch.float32, device=ALAT)
    for _ in range(EPOCH):
        m.train()
        perm = rng.permutation(fit)
        for i in range(0, len(perm), BATCH):
            xb, zb, r = ambil(W, Z, perm[i:i + BATCH], sd)
            opt.zero_grad()
            loss(m(xb, zb), yt[r]).backward()
            opt.step()
    m.eval()
    out = np.zeros(len(uji))
    with torch.no_grad():
        for i in range(0, len(uji), 512):
            blok = uji[i:i + 512]
            urut = np.argsort(blok)
            xb, zb, _ = ambil(W, Z, blok, sd)
            out[i + urut] = m(xb, zb).float().cpu().numpy()
    return out


def main():
    s, n = sys.argv[1], int(sys.argv[2])
    d = pd.read_csv(os.path.join(T, 'kohort.csv'))
    y, g = d[s].values.astype(int), d['patient_id'].values
    W = np.load(os.path.join(T, 'gelombang12.npy'), mmap_mode='r')
    Z = np.stack([d['age'].values / 100, d['pria'].values], 1).astype(np.float32)
    print(f'{s}: {len(y)} rekaman, {y.sum()} positif; gelombang {W.shape}; alat {ALAT}', flush=True)
    for b in range(n):
        f = os.path.join(T, f'oof_{s}_cnn_benih{b}.npy')
        if os.path.exists(f):
            print(f'  benih {b}: dari berkas', flush=True)
            continue
        P = np.zeros(len(y))
        for k, (tr, te) in enumerate(StratifiedGroupKFold(LIPAT, shuffle=True, random_state=b).split(Z, y, g)):
            fk = os.path.join(T, f'oof_{s}_cnn_benih{b}_lipat{k}.npy')
            if os.path.exists(fk):
                P[te] = np.load(fk)
                continue
            t0 = time.time()
            rng = np.random.default_rng(1000 * b + k)
            ug = np.unique(g[tr])
            cal_g = set(rng.choice(ug, len(ug) // 10, replace=False))
            cal = np.array([x in cal_g for x in g[tr]])
            fit, cl = tr[~cal], tr[cal]
            out = latih_prediksi(W, Z, y, fit, np.concatenate([cl, te]), b)
            lr = LogisticRegression(C=1e6, max_iter=1000).fit(out[:len(cl), None], y[cl])
            P[te] = lr.predict_proba(out[len(cl):, None])[:, 1]
            np.save(fk, P[te])
            print(f'  benih {b} lipat {k}: AUROC {roc_auc_score(y[te], P[te]):.3f} ({time.time() - t0:.0f} d)', flush=True)
        np.save(f, P)
        print(f'  benih {b}: AUROC {roc_auc_score(y, P):.3f}', flush=True)


if __name__ == '__main__':
    main()
