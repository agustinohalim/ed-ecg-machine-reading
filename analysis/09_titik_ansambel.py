"""Nightingale ed-bwh-ecg — AUROC dan AUPRC titik untuk prediksi ansambel (rata-rata lima benih), kedua sasaran.

06_cnn12.py mencetak AUROC rerata per benih dan CI95 ansambel, tetapi tidak AUROC ansambel itu sendiri. Skrip
ini membaca prediksi OOF tersimpan dan mencetak AUROC/AUPRC ansambel serta AUROC tiap benih, tanpa melatih apa
pun.

Pakai: python 09_titik_ansambel.py > titik_ansambel.log
"""
import importlib.util
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

D = os.path.join(os.environ.get('NGSCI_DATA', 'C:/Data/Nightingale'), 'ed-bwh-ecg')
T = os.path.join(D, 'turunan')
spek = importlib.util.spec_from_file_location('gd', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                 '01_garis_dasar.py'))
gd = importlib.util.module_from_spec(spek)
spek.loader.exec_module(gd)


def main():
    d = gd.kohort()
    for s in ['macetrop_pos_or_death_030', 'death_030_day']:
        y = d[s].values
        print(f'== {s}: {len(y)} kunjungan, {int(y.sum())} kejadian')
        for n in ['M0', 'M1l', 'M1g', 'M2', 'M3', 'M4', 'M5']:
            f = os.path.join(T, f'oof_{s}_{n}.npy')
            if not os.path.exists(f):
                continue
            P = np.load(f)
            p = P.mean(0)
            print(f'  {n:4} AUROC {roc_auc_score(y, p):.3f}  AUPRC {average_precision_score(y, p):.3f}  per benih '
                  + ' '.join(f'{roc_auc_score(y, q):.3f}' for q in P))


if __name__ == '__main__':
    main()
