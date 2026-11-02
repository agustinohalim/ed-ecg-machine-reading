"""Nightingale ed-bwh-ecg — Gambar 3: AUROC dan selisih CNN-12 − Report menurut ukuran ansambel.

Angka dibaca dari ukuran_ansambel.log (15_ukuran_ansambel.py), yang hanya memuat agregat, jadi gambar boleh masuk repo.
(a) AUROC Report dan CNN-12 menurut k; (b) selisih CNN-12 − Report dengan CI95 pasien + benih, garis nol dan margin
±0,02. Warna sama dengan 04_gambar.py.

Pakai: python 16_gambar_kurva.py
"""
import os
import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

H = os.path.dirname(os.path.abspath(__file__))
G = os.path.join(H, 'gambar')
POLA = re.compile(r'k=\s*(\d+): M4 ([\d.]+) M1g ([\d.]+) selisih ([+-][\d.]+) \[([+-][\d.]+); ([+-][\d.]+)\]')


def main():
    baris = [POLA.search(t) for t in open(os.path.join(H, 'ukuran_ansambel.log'), encoding='utf-8')]
    r = [[float(x) for x in m.groups()] for m in baris if m]
    k, m4, m1, d, lo, hi = (list(c) for c in zip(*r))
    plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans'})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.2))

    a.plot(k, m1, color='#1f77b4', lw=2, marker='o', ms=4)
    a.plot(k, m4, color='#d62728', lw=2, marker='s', ms=4)
    a.text(7, m1[4] - 0.0025, 'Report', color='#333333', ha='center', va='top')
    a.text(4.2, m4[3] + 0.0035, 'CNN-12', color='#333333', ha='right', va='bottom')
    a.set_xscale('log')
    a.set_xticks(k, [str(int(x)) for x in k])
    a.set_xlabel('Runs averaged (k)')
    a.set_ylabel('AUROC')
    a.set_title('(a) AUROC by number of runs averaged', loc='left', fontsize=9)

    b.axhspan(-0.02, 0.02, color='#eeeeee', lw=0)
    b.axhline(0, color='#777777', lw=0.8)
    b.fill_between(k, lo, hi, color='#d62728', alpha=0.18, lw=0)
    b.plot(k, d, color='#d62728', lw=2, marker='s', ms=4)
    b.text(k[0] * 1.05, 0.0205, 'margin ±0.02', color='#555555', va='bottom', fontsize=8)
    b.set_xscale('log')
    b.set_xticks(k, [str(int(x)) for x in k])
    b.set_xlabel('Runs averaged (k)')
    b.set_ylabel('CNN-12 − Report, AUROC')
    b.set_title('(b) Difference with 95 % interval', loc='left', fontsize=9)

    for ax in (a, b):
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#e5e5e5', lw=0.6)
        ax.minorticks_off()
    fig.tight_layout()
    os.makedirs(G, exist_ok=True)
    fig.savefig(os.path.join(G, 'gambar3_ukuran_ansambel.png'), dpi=300)


if __name__ == '__main__':
    main()
