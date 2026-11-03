<!-- Copy of the pre-registration as committed on 2 Nov 2026 (commit 5acb96e in the authors' private working repository), before PTB-XL data were downloaded and before screening. Indonesian; unchanged. -->

## Prapendaftaran: replikasi PTB-XL+ dan audit pustaka (ditulis 2 Nov 2026, sebelum data diunduh dan sebelum kode)

Disetujui pengguna 2 Nov 2026 ("A+B go"). Bagian ini dikomit sebelum ada angka; jangan disunting sesudah hasil keluar.
Perubahan sesudahnya ditulis sebagai catatan bertanggal di bawahnya, beserta alasannya.

**A. Replikasi kurva ukuran ansambel di PTB-XL+** (PhysioNet, CC BY 4.0, terbuka): PTB-XL 1.0.3 (21.799 EKG, 18.869
pasien, 12 sadapan 10 detik 500 Hz) + PTB-XL+ 1.0.1 (ukuran dan pernyataan GE Marquette 12SL).
- Himpunan: rekaman dengan ≥ 1 label superkelas diagnostik (pemetaan `scp_statements.csv`, kode diagnostik apa pun
  tanpa ambang likelihood, seperti tolok ukur Strodthoff dkk.) dan keluaran 12SL tersedia.
- Sasaran utama: MI (ya/tidak). Sekunder: STTC, CD, HYP.
- Report: HistGB (pengaturan M1g: max_iter 300, learning_rate 0,05, random_state 0) atas delapan ukuran global 12SL
  yang setara dengan ED-BWH (sumbu P, R, T; PR; QRS; QT; QTc; laju) bila ada, pernyataan 12SL berprevalensi ≥ 1 %
  sebagai bendera, usia, jenis kelamin. Analisis kepekaan: semua ukuran 12SL.
- CNN-12: arsitektur M4 (lima blok Conv1d k7, kanal 32-64-64-128-128, rerata global, usia dan jenis kelamin
  disambung, dropout 0,3, 15 epoch, Adam 1e-3, batch 128, pos_weight, kalibrasi ulang logistik pada 10 % pasien
  lipatan latih). Satu-satunya perbedaan: 12 kanal masukan penuh 10 detik (ED-BWH: susunan kertas 15 baris).
- Putaran: tiap putaran b = StratifiedGroupKFold 5 lipatan (grup = pasien, random_state b) + inisialisasi baru,
  seperti ED-BWH; lipatan resmi `strat_fold` tidak dipakai karena tetap (tidak bisa diacak per putaran).
  MI: 25 putaran. Sekunder: 10 putaran tiap sasaran (waktu GPU), kurva hanya sampai k = 10.
- Bacaan: kurva k = 1, 2, 3, 5, 7, 10, (15, 25 untuk MI); bootstrap pasien 1.000, benih 20261104; aturan baca sama
  dengan naskah (95 % memisahkan 0 → ≠0; margin ±0,02 pada 90 %). Peringatan kolam terbatas berlaku sama.
- Yang dilaporkan apa pun hasilnya: CNN-12 − Report pada k = 1 dan k = 25 (MI) / k = 10 (sekunder), dan
  (selisih)_k − (selisih)_1. Hasil "tidak bergeser" dilaporkan sama jelasnya dengan "bergeser".
- Label PTB-XL berasal dari kardiolog dan alat Schiller, bukan dari 12SL; dicatat sebagai beda dari ED-BWH.

**B. Audit pelaporan putaran dalam pustaka deep learning EKG**
- PubMed, 1 Jan 2023 – 31 Okt 2026, artikel jurnal berbahasa Inggris:
  `(electrocardiogra*[tiab] OR ECG[tiab]) AND ("deep learning"[tiab] OR "convolutional neural network"[tiab] OR
  "neural network"[tiab])`.
- Urutan acak seluruh PMID dengan benih 20261104; disaring berurutan sampai 60 studi masuk dengan teks lengkap
  (PMC). Masuk: studi asli yang melatih jaringan dalam atas gelombang EKG untuk diagnosis atau luaran dan
  melaporkan metrik kinerja. Yang tanpa teks lengkap dicatat jumlahnya, tidak dikode.
- Kode per studi: jumlah putaran/benih dinyatakan (angka / tidak dinyatakan); variasi antarputaran dilaporkan (ya/
  tidak); ansambel putaran (ya/tidak/tidak jelas); pembanding non-deep-learning (ya/tidak, apa); selang
  kepercayaan metrik (jenis); kode tersedia (ya/tidak).
- Luaran utama: proporsi studi yang tidak menyatakan jumlah putaran, dengan selang Wilson 95 %.
