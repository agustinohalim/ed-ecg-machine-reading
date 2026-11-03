"""Ringkasan audit pustaka (prapendaftaran B, Hasil_Uji_EDBWH.md).

Membaca kode_60.csv di folder yang sama dan mencetak hitungan, proporsi, dan
selang Wilson 95 % untuk luaran yang diprapendaftarkan. Hanya pustaka standar.
"""
import csv
import math
import os

Z = 1.959963984540054


def wilson(k, n, z=Z):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def baris(nama, k, n):
    lo, hi = wilson(k, n)
    print(f"{nama:<52} {k:>3}/{n:<3} = {k / n:6.1%}  (Wilson 95%: {lo:6.1%} - {hi:6.1%})")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "kode_60.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    tidak = [r for r in rows if r["runs_stated"].strip().lower() == "not stated"]
    var = [r for r in rows if r["run_variability_reported"].strip().lower() == "yes"]
    ens = [r for r in rows if r["run_ensemble"].strip().lower() == "yes"]
    ens_unclear = [r for r in rows if r["run_ensemble"].strip().lower() == "unclear"]
    nondl = [r for r in rows if r["non_dl_comparator"].strip().lower().startswith("yes")]
    nondl_tidak = [r for r in nondl if r["runs_stated"].strip().lower() == "not stated"]
    kode = [r for r in rows if r["code_available"].strip().lower() == "yes"]

    print(f"Studi terkode: {n}")
    print()
    baris("Jumlah putaran tidak dinyatakan (luaran utama)", len(tidak), n)
    baris("Variasi antarputaran dilaporkan", len(var), n)
    baris("Ansambel putaran dipakai (ya)", len(ens), n)
    baris("  ansambel putaran tidak jelas (unclear)", len(ens_unclear), n)
    baris("Ada pembanding non-deep-learning", len(nondl), n)
    baris("Di antara yang ada pembanding non-DL: putaran tak dinyatakan", len(nondl_tidak), len(nondl))
    baris("Kode tersedia (tambahan, deskriptif)", len(kode), n)
    print()
    stated = [r for r in rows if r not in tidak]
    print("Jumlah putaran yang dinyatakan:",
          ", ".join(f"PMID {r['pmid']}={r['runs_stated']}" for r in stated))
    ci = {}
    for r in rows:
        t = r["ci_type"].lower()
        key = ("bootstrap" if "bootstrap" in t else
               "DeLong" if "delong" in t else
               "none" if t.startswith("none") else
               "method not stated" if "not stated" in t else "other")
        ci[key] = ci.get(key, 0) + 1
    print("Jenis selang kepercayaan (kategori utama):",
          ", ".join(f"{k}={v}" for k, v in sorted(ci.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
