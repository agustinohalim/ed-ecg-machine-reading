"""Nightingale ed-bwh-ecg — Tabel 1 naskah: karakteristik kohort menurut sasaran utama.

Kohort dari 01_garis_dasar.py (turunan/kohort.csv), ras dan pendapatan dari patient.csv, bacaan mesin
dari ecg-metadata.csv (EKG pertama yang sama). Hanya agregat yang dicetak.

Pakai: python 05_tabel1.py > tabel1.log
"""
import os

import numpy as np
import pandas as pd

D = os.path.join(os.environ.get('NGSCI_DATA', 'C:/Data/Nightingale'), 'ed-bwh-ecg')
T = os.path.join(D, 'turunan')
S = 'macetrop_pos_or_death_030'
BENDERA = ['has_normal_ecg', 'has_afib', 'has_bbb', 'has_lvh', 'has_st_eleva', 'has_depress',
           'has_twave_inver', 'has_prolonged_qt', 'has_pacemaker']


def main():
    d = pd.read_csv(os.path.join(T, 'kohort.csv'))
    d = d.merge(pd.read_csv(os.path.join(D, 'v1', 'patient.csv')), on='patient_ngsci_id', how='left')
    m = pd.read_csv(os.path.join(D, 'v1', 'ecg-metadata.csv'), usecols=['ecg_id', 'vent_rate', 'qrs_duration',
                                                                        'qtc_interval'] + BENDERA)
    d = d.merge(m, on='ecg_id', how='left', validate='one_to_one')
    kel = {'all': d, 'event': d[d[S] == 1], 'no event': d[d[S] == 0]}
    print(f"{'':28}" + ''.join(f'{k:>22}' for k in kel))
    print(f"{'visits':28}" + ''.join(f'{len(x):>22,}' for x in kel.values()))
    print(f"{'patients':28}" + ''.join(f'{x.patient_ngsci_id.nunique():>22,}' for x in kel.values()))

    def baris(nama, f):
        print(f'{nama:28}' + ''.join(f'{f(x):>22}' for x in kel.values()))

    baris('age, mean (SD)', lambda x: f'{x.age_at_admit.mean():.1f} ({x.age_at_admit.std():.1f})')
    baris('age >= 65, n (%)', lambda x: f'{(x.age_at_admit >= 65).sum():,} ({(x.age_at_admit >= 65).mean():.1%})')
    baris('male, n (%)', lambda x: f'{int(x.pria.sum()):,} ({x.pria.mean():.1%})')
    for r in ['black', 'hispanic', 'white', 'other']:
        baris(f'race {r}, n (%)', lambda x, r=r: f"{int(x[f'race_{r}'].sum()):,} ({x[f'race_{r}'].mean():.1%})")
    baris('agi<25k share, median', lambda x: f'{x.agi_under_25k.median():.3f}')
    baris('tested <=10 d, n (%)', lambda x: f'{int(x.test_010_day.sum()):,} ({x.test_010_day.mean():.1%})')
    baris('death 30 d, n (%)', lambda x: f'{int(x.death_030_day.sum()):,} ({x.death_030_day.mean():.1%})')
    for c in ['vent_rate', 'qrs_duration', 'qtc_interval']:
        baris(f'{c}, median (IQR)', lambda x, c=c: f'{x[c].median():.0f} ({x[c].quantile(.25):.0f}-{x[c].quantile(.75):.0f})')
    for c in BENDERA:
        baris(f'{c}, n (%)', lambda x, c=c: f'{int(x[c].sum()):,} ({x[c].mean():.1%})')
    print(f"\nnilai hilang: {d[['vent_rate', 'qrs_duration', 'qtc_interval']].isna().sum().to_dict()}; "
          f'agi {d.agi_under_25k.isna().sum()}')


if __name__ == '__main__':
    main()
