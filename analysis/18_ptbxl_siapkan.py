"""PTB-XL+ — kohort, ciri bacaan mesin 12SL, dan gelombang 12 sadapan (prapendaftaran 2 Nov 2026, Hasil bagian A).

Data terbuka PhysioNet (CC BY 4.0): PTB-XL 1.0.3 dan PTB-XL+ 1.0.1, diekstrak di PTBXL_DATA (bawaan
C:/Data/PhysioNet/ptbxl: xl/<folder>/ dan plus/<folder>/). Keluaran ke PTBXL_DATA/turunan, bukan ke repo.

Kohort: rekaman dengan >= 1 superkelas diagnostik (kode diagnostik apa pun di scp_codes, tanpa ambang likelihood,
dipetakan lewat scp_statements.csv) dan baris 12SL. Usia 300 (dianonimkan, > 89) dijadikan 90. Jenis kelamin PTB-XL:
0 = laki-laki.

Ciri Report = delapan ukuran global 12SL setara ED-BWH (sumbu P, R, T; PR; QRS; QT; QTc; laju ventrikel) + bendera
pernyataan 12SL berprevalensi >= 1 % dalam kohort + usia + jenis kelamin. Ciri kepekaan = semua kolom 12sl_features.

Pakai: python 18_ptbxl_siapkan.py > ptbxl_siapkan.log
"""
import ast
import glob
import os
import time

import numpy as np
import pandas as pd
import wfdb

B = os.environ.get('PTBXL_DATA', 'C:/Data/PhysioNet/ptbxl')
XL = glob.glob(os.path.join(B, 'xl', '*'))[0]
PL = glob.glob(os.path.join(B, 'plus', '*'))[0]
T = os.path.join(B, 'turunan')
GLOBAL8 = ['P_AxisFront_Global', 'R_AxisFrontal_Global', 'T_AxisFront_Global', 'PR_Int_Global',
           'QRS_Dur_Global', 'QT_Int_Global', 'QT_IntCorr_Global', 'HR_Ventr_Global']
SUPER = ['MI', 'STTC', 'CD', 'HYP', 'NORM']


def main():
    os.makedirs(T, exist_ok=True)
    db = pd.read_csv(os.path.join(XL, 'ptbxl_database.csv'))
    scp = pd.read_csv(os.path.join(XL, 'scp_statements.csv'), index_col=0)
    kelas = scp[scp.diagnostic == 1].diagnostic_class.to_dict()
    kode = db.scp_codes.apply(ast.literal_eval)
    sup = kode.apply(lambda c: {kelas[k] for k in c if k in kelas})
    for s in SUPER:
        db[s] = sup.apply(lambda x, s=s: int(s in x))
    f12 = pd.read_csv(os.path.join(PL, 'features', '12sl_features.csv'))
    st = pd.read_csv(os.path.join(PL, 'labels', '12sl_statements.csv'))
    st['daftar'] = st.statements.apply(ast.literal_eval)
    print(f'PTB-XL {len(db)} rekaman; 12SL ciri {len(f12)}, pernyataan {len(st)}')
    d = db[sup.apply(len) > 0].merge(f12[['ecg_id']], on='ecg_id').merge(st[['ecg_id', 'daftar']], on='ecg_id')
    d = d.sort_values('ecg_id').reset_index(drop=True)
    d['age'] = d.age.clip(upper=90)
    d['pria'] = (d.sex == 0).astype(int)
    d['patient_id'] = d.patient_id.astype(int)
    print(f'kohort: {len(d)} rekaman, {d.patient_id.nunique()} pasien '
          f'(tanpa superkelas {int((sup.apply(len) == 0).sum())})')
    for s in SUPER:
        print(f'  {s}: {d[s].sum()} positif ({100 * d[s].mean():.1f} %)')

    semua = sorted({x for L in d.daftar for x in L})
    prev = {x: np.mean([x in L for L in d.daftar]) for x in semua}
    pilih = [x for x in semua if prev[x] >= 0.01]
    print(f'pernyataan 12SL: {len(semua)} jenis, {len(pilih)} berprevalensi >= 1 %: {pilih}')
    F = f12.set_index('ecg_id').loc[d.ecg_id]
    hilang = F[GLOBAL8].isna().mean()
    print('nilai hilang ukuran global: ' + ', '.join(f'{k} {100 * v:.1f} %' for k, v in hilang.items()))
    R = pd.DataFrame({'ecg_id': d.ecg_id.values, 'age': d.age.values, 'pria': d.pria.values})
    for k in GLOBAL8:
        R[k] = F[k].values
    for x in pilih:
        R['st_' + x] = [int(x in L) for L in d.daftar]
    R.to_csv(os.path.join(T, 'ciri_report.csv'), index=False)
    S = F.reset_index(drop=True)
    S.insert(0, 'ecg_id', d.ecg_id.values)
    S['age'], S['pria'] = d.age.values, d.pria.values
    S.to_csv(os.path.join(T, 'ciri_12sl_semua.csv'), index=False)
    d[['ecg_id', 'patient_id', 'age', 'pria', 'strat_fold', 'filename_hr'] + SUPER].to_csv(
        os.path.join(T, 'kohort.csv'), index=False)
    print(f'ciri Report: {R.shape[1] - 1} kolom; ciri semua 12SL: {S.shape[1] - 1} kolom')

    f = os.path.join(T, 'gelombang12.npy')
    if os.path.exists(f):
        print('gelombang: dari berkas')
        return
    t0 = time.time()
    W = np.lib.format.open_memmap(f + '.tmp', mode='w+', dtype=np.float16, shape=(len(d), 12, 5000))
    nan = 0
    for i, fn in enumerate(d.filename_hr):
        x = wfdb.rdsamp(os.path.join(XL, fn))[0].T[:, :5000]
        nan += int(np.isnan(x).sum())
        x = np.nan_to_num(x - np.nanmean(x, axis=1, keepdims=True))
        W[i, :, :x.shape[1]] = x.astype(np.float16)
        if i % 5000 == 0:
            print(f'  gelombang {i}/{len(d)}, {time.time() - t0:.0f} d', flush=True)
    W.flush()
    del W
    os.replace(f + '.tmp', f)
    print(f'gelombang selesai: {len(d)} × 12 × 5000, nilai NaN {nan}, {time.time() - t0:.0f} d')


if __name__ == '__main__':
    main()
