"""Nightingale ed-bwh-ecg — M4: CNN 1D atas 12 sadapan 500 Hz; M5: gabungan M1g + M4.

Menjawab telaah 28 Okt 2026 (Telaah/2026-10-28-EDBWH): M2 hanya melihat V1, II, V5 pada 100 Hz,
sedangkan bacaan mesin membaca 12 sadapan. Berkas npz dataset menyimpan tiap EKG sebagai 15 × 5.000
(500 Hz, 10 detik): baris 0–11 = I, II, III, aVR, aVL, aVF, V1–V6, masing-masing hanya 2,5 detik pada
posisi waktunya di kertas (sisanya NaN); baris 12–14 = V1, II, V5 penuh 10 detik. Masukan M4 adalah
susunan itu apa adanya: rerata tiap baris dikurangkan, NaN menjadi 0, disimpan float16 di folder
turunan (C:/Data/Nightingale/ed-bwh-ecg/turunan/gelombang15.npy), dibaca langsung dari zip dataset.

Arsitektur = M2 dengan 15 kanal masukan dan satu blok tambahan untuk panjang 5.000: lima blok
Conv1d(kernel 7) + BatchNorm + ReLU + MaxPool(4), kanal 32-64-64-128-128, rerata global, usia/100 dan
jenis kelamin disambung, dropout 0,3. Pelatihan, lipatan, kalibrasi ulang 10 % pasien, M5, dan
bootstrap sama dengan 02_cnn.py. Berjalan di GPU (venv-gpu).

Pakai: venv-gpu python 06_cnn12.py [sasaran] > cnn12.log
"""
import io
import os
import sys
import time
import zipfile

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
ALAT = 'cuda' if torch.cuda.is_available() else 'cpu'


def siapkan(d):
    """Gelombang 15 × 5.000 untuk tiap kunjungan kohort, urutan baris = kohort.csv."""
    f = os.path.join(T, 'gelombang15.npy')
    if os.path.exists(f):
        return np.load(f, mmap_mode='r')
    t0 = time.time()
    W = np.lib.format.open_memmap(f + '.tmp', mode='w+', dtype=np.float16, shape=(len(d), 15, 5000))
    panjang, kosong = {}, 0
    with zipfile.ZipFile(os.path.join(D, 'v1', 'ecg-waveforms-npz.zip')) as z:
        nama = {os.path.basename(n)[:-4]: n for n in z.namelist() if n.endswith('.npz')}
        for i, e in enumerate(d['ecg_id'].str.replace('ecg', '', n=1)):
            x = np.load(io.BytesIO(z.read(nama[e])))['waveform']
            panjang[x.shape] = panjang.get(x.shape, 0) + 1
            kosong += int(np.isnan(x).all(1).sum())
            with np.errstate(all='ignore'):
                x = x - np.nanmean(x, axis=1, keepdims=True)
            # sebagian kecil berkas lebih panjang atau lebih pendek dari 5.000: dipotong atau diisi 0
            y = np.zeros((15, 5000))
            y[:, :min(5000, x.shape[1])] = np.nan_to_num(x[:15, :5000])
            W[i] = y.astype(np.float16)
            if i % 5000 == 0:
                print(f'  siapkan {i}/{len(d)}, {time.time() - t0:.0f} d', flush=True)
    print(f'  bentuk berkas: {panjang}; sadapan kosong seluruhnya: {kosong}', flush=True)
    W.flush()
    del W
    os.replace(f + '.tmp', f)
    return np.load(f, mmap_mode='r')


class Jaring(nn.Module):
    def __init__(self):
        super().__init__()
        lap, c0 = [], 15
        for c in [32, 64, 64, 128, 128]:
            lap += [nn.Conv1d(c0, c, 7, padding=3), nn.BatchNorm1d(c), nn.ReLU(), nn.MaxPool1d(4)]
            c0 = c
        self.fitur = nn.Sequential(*lap)
        self.kepala = nn.Sequential(nn.Dropout(0.3), nn.Linear(128 + 2, 1))

    def forward(self, x, z):
        return self.kepala(torch.cat([self.fitur(x).mean(-1), z], 1)).squeeze(-1)


def ambil(W, Z, r, sd):
    r = np.sort(r)  # baca berurutan dari memmap
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


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def oof_cnn(W, Z, y, g, s):
    P = np.zeros((len(BENIH), len(y)))
    for i, b in enumerate(BENIH):
        f = os.path.join(T, f'oof_{s}_M4_benih{b}.npy')
        if os.path.exists(f):
            P[i] = np.load(f)
            print(f'  benih {b}: dari berkas', flush=True)
            continue
        for k, (tr, te) in enumerate(StratifiedGroupKFold(LIPAT, shuffle=True, random_state=b).split(Z, y, g)):
            # tiap lipatan juga disimpan: proses pernah dihentikan karena memori sistem menipis
            fk = os.path.join(T, f'oof_{s}_M4_benih{b}_lipat{k}.npy')
            if os.path.exists(fk):
                P[i, te] = np.load(fk)
                print(f'  benih {b} lipat {k}: dari berkas', flush=True)
                continue
            t0 = time.time()
            rng = np.random.default_rng(1000 * b + k)
            ug = np.unique(g[tr])
            cal_g = set(rng.choice(ug, len(ug) // 10, replace=False))
            cal = np.array([x in cal_g for x in g[tr]])
            fit, cl = tr[~cal], tr[cal]
            out = latih_prediksi(W, Z, y, fit, np.concatenate([cl, te]), b)
            lr = LogisticRegression(C=1e6, max_iter=1000).fit(out[:len(cl), None], y[cl])
            P[i, te] = lr.predict_proba(out[len(cl):, None])[:, 1]
            np.save(fk, P[i, te])
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
    W = siapkan(d)
    Z = np.stack([d['age_at_admit'].values / 100, d['pria'].values], 1).astype(np.float32)
    print(f'{s}: {len(y)} kunjungan, {y.sum()} kejadian; gelombang {W.shape}; alat {ALAT}', flush=True)
    f = os.path.join(T, f'oof_{s}_M4.npy')
    if os.path.exists(f):
        P4 = np.load(f)
    else:
        P4 = oof_cnn(W, Z, y, g, s)
        np.save(f, P4)
    P0, P1, P2 = (np.load(os.path.join(T, f'oof_{s}_{n}.npy')) for n in ('M0', 'M1g', 'M2'))
    P5 = np.zeros_like(P4)
    for i, b in enumerate(BENIH):
        X = np.stack([logit(P1[i]), logit(P4[i])], 1)
        for tr, te in StratifiedGroupKFold(LIPAT, shuffle=True, random_state=100 + b).split(X, y, g):
            P5[i, te] = LogisticRegression(max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    np.save(os.path.join(T, f'oof_{s}_M5.npy'), P5)
    semua = {'M0': P0, 'M1g': P1, 'M2': P2, 'M4': P4, 'M5': P5}
    rata = {n: P.mean(0) for n, P in semua.items()}
    bs = boot(y, rata, g, np.random.default_rng(20261028))
    print(f"\n{'model':5} {'AUROC (SD benih)':18} {'CI95':15} {'AUPRC (SD)':16} {'CI95':15} {'Brier':7} "
          f"slope  slope per benih")
    for n, P in semua.items():
        au = [roc_auc_score(y, p) for p in P]
        ap = [average_precision_score(y, p) for p in P]
        ca = np.percentile([h[n][0] for h in bs], [2.5, 97.5])
        cp = np.percentile([h[n][1] for h in bs], [2.5, 97.5])
        sb = [slope(y, p) for p in P]
        print(f'{n:5} {np.mean(au):.3f} ({np.std(au):.3f})    [{ca[0]:.3f}, {ca[1]:.3f}]  '
              f'{np.mean(ap):.3f} ({np.std(ap):.3f})  [{cp[0]:.3f}, {cp[1]:.3f}]  '
              f'{brier_score_loss(y, rata[n]):.4f}  {slope(y, rata[n]):.2f}   {np.mean(sb):.2f} ± {np.std(sb):.2f}')
    print('\nΔAUROC berpasangan (prediksi rata-rata benih), CI95 bootstrap pasien')
    for a, b in [('M4', 'M0'), ('M4', 'M2'), ('M4', 'M1g'), ('M5', 'M1g')]:
        dlt = roc_auc_score(y, rata[a]) - roc_auc_score(y, rata[b])
        ci = np.percentile([h[a][0] - h[b][0] for h in bs], [2.5, 97.5])
        print(f'{a} − {b}: {dlt:+.3f}  [{ci[0]:+.3f}, {ci[1]:+.3f}]')


if __name__ == '__main__':
    main()
