"""PTB-XL+ — ciri Report dari pembuat lain (Uni-G, University of Glasgow), analisis pasca-hasil (Hasil bagian C, 3 Nov 2026).

Menulis tiga berkas ciri ke PTBXL_DATA/turunan, urutan baris sama dengan kohort.csv:
  ciri_unig8.csv      delapan ukuran global Uni-G padanan Report utama + usia + jenis kelamin
  ciri_unig_semua.csv semua kolom unig_features.csv + usia + jenis kelamin
  ciri_sl8.csv        delapan ukuran global 12SL yang sama + usia + jenis kelamin (jangkar sepembuat)
Rekaman kohort tanpa baris Uni-G dibiarkan bernilai hilang.

Pakai: python 22_ptbxl_unig.py > ptbxl_unig.log
"""
import glob
import os

import pandas as pd

B = os.environ.get('PTBXL_DATA', 'C:/Data/PhysioNet/ptbxl')
PL = glob.glob(os.path.join(B, 'plus', '*'))[0]
T = os.path.join(B, 'turunan')
SL8 = ['P_AxisFront_Global', 'R_AxisFrontal_Global', 'T_AxisFront_Global', 'PR_Int_Global',
       'QRS_Dur_Global', 'QT_Int_Global', 'QT_IntCorr_Global', 'HR_Ventr_Global']
UG8 = ['P_AxisFront_Global', 'QRS_AxisFront_Global', 'T_AxisFront_Global', 'PR_Int_Global',
       'QRS_Dur_Global', 'QT_Int_Global', 'QT_IntCorr_Global', 'HR_Ventr_Global']


def main():
    d = pd.read_csv(os.path.join(T, 'kohort.csv'))
    u = pd.read_csv(os.path.join(PL, 'features', 'unig_features.csv')).set_index('ecg_id')
    s = pd.read_csv(os.path.join(T, 'ciri_12sl_semua.csv'))
    assert (s.ecg_id.values == d.ecg_id.values).all()
    ada = d.ecg_id.isin(u.index)
    print(f'kohort {len(d)}; dengan baris Uni-G {int(ada.sum())}, tanpa {int((~ada).sum())}')
    U = u.reindex(d.ecg_id.values)
    print('nilai hilang ukuran global Uni-G: ' + ', '.join(f'{k} {100 * U[k].isna().mean():.1f} %' for k in UG8))
    dem = pd.DataFrame({'ecg_id': d.ecg_id.values, 'age': d.age.values, 'pria': d.pria.values})
    pd.concat([dem, U[UG8].reset_index(drop=True)], axis=1).to_csv(os.path.join(T, 'ciri_unig8.csv'), index=False)
    pd.concat([dem, U.reset_index(drop=True)], axis=1).to_csv(os.path.join(T, 'ciri_unig_semua.csv'), index=False)
    pd.concat([dem, s[SL8]], axis=1).to_csv(os.path.join(T, 'ciri_sl8.csv'), index=False)
    print(f'ciri unig8 {len(UG8) + 2}, unig_semua {U.shape[1] + 2}, sl8 {len(SL8) + 2} kolom')


if __name__ == '__main__':
    main()
