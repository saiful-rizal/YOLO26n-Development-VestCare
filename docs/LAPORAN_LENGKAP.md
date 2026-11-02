# Laporan Lengkap — VestCare YOLO26n-seg (Tahap 6–8)

Dokumen ini merangkum seluruh eksperimen, angka akurasi, indikator hasil, metode,
keterbatasan, dan cara reproduksi. Ringkasan singkat ada di [`README.md`](../README.md);
komposisi dataset ada di [`DATASET.md`](DATASET.md); semua gambar ada di
[`GALERI.md`](GALERI.md).

---

## 1. Tujuan dan ruang lingkup

Proyek ini mengembangkan model segmentasi objek jalan untuk rompi tunanetra
**VestCare**. Model harus mengenali lima kelas: **pohon, lubang, orang, motor,
mobil** pada gambar jalan.

Ruang lingkup yang dilaporkan:

| Tahap | Fokus | Keluaran |
|---|---|---|
| 6 | Bandingkan `yolo11n-seg` vs `yolo26n-seg` pada 3 seed split | model pemenang + bukti keunggulan |
| 7 | Latih ulang model pemenang pada train+val, 100 epoch | `models/best.pt` + skor test asli |
| 8 | Evaluasi mendalam model final | per kelas, per varian, tingkat gambar, kesalahan, kecepatan |
| 9 (usulan) | Fuzzy Sugeno untuk urgensi bahaya | skrip simulasi, belum dievaluasi kuantitatif |

Semua angka di laporan ini berasal dari file hasil di repo. Tidak ada angka yang
diisi manual tanpa sumber.

---

## 2. Dataset

Split resmi yang dipakai: `split_r80_s44` (rasio 80, seed 44).

| Split | Gambar | Instance |
|---|---:|---:|
| Train | 3.060 | 6.465 |
| Val | 770 | 1.440 |
| Test | 430 | 905 |
| **Total** | **4.260** | **8.810** |

Test set terdiri dari lima varian seimbang: `asli`, `gerak_diagonal`,
`gerak_horizontal`, `gerak_vertikal`, `getar` — masing-masing **86 gambar**.
Rincian lengkap per kelas dan per varian: [`DATASET.md`](DATASET.md).

Pada tahap 7, train + val digabung lalu dipisah lagi menjadi **3.405 gambar train
final** dan **425 gambar val-internal** (kelompok varian/frame tidak dipisah).
Test asli 430 gambar tidak pernah dipakai untuk latih/val.

---

## 3. Metodologi

### 3.1 Tahap 6 — perbandingan model

- 2 model: `yolo11n-seg`, `yolo26n-seg` (bobot pralatih COCO).
- 3 seed split: 43, 44, 46.
- 6 pelatihan, masing-masing 10 epoch, `patience` 30, `imgsz` 416, `batch` 1,
  seed latih 0, `deterministic=True`.
- Optimizer memakai resep bawaan Ultralytics (`auto`).
- Evaluasi pada val dan test dengan `model.val()`.
- Pemilihan model/rasio memakai skor **test** (bukan val).

### 3.2 Tahap 7 — latih final

- Model terpilih: `yolo26n-seg`; rasio 80; seed split 44.
- Train + val digabung; val-internal 10% dipilih per kelompok.
- Latih ulang dari bobot COCO, **100 epoch**, `patience` 30, `imgsz` 416,
  `batch` 1, seed latih 0, `deterministic=True`.
- Augmentasi bawaan Ultralytics: mosaic, HSV, translate, scale, fliplr, erasing;
  `close_mosaic=10`.
- Model dievaluasi pada test asli, lalu bobot terbaik disalin ke `models/best.pt`.

### 3.3 Tahap 8 — evaluasi mendalam

- Model: `models/best.pt`; split: `split_r80_s44`; 430 gambar test.
- `model.val()` untuk metrik mask dan kotak.
- Evaluasi per varian dengan membentuk subset test sementara.
- Analisis tingkat gambar: `model.predict(conf=0.25)`, lalu bandingkan
  **himpunan kelas** yang diprediksi dengan himpunan kelas ground truth.
- Contoh prediksi benar/salah disimpan (maksimal 20 per folder).
- Kecepatan diukur pada 100 gambar, batch 1, termasuk pra/pasca-proses.

### 3.4 Definisi metrik

| Metrik | Definisi |
|---|---|
| Precision (P) | TP / (TP + FP) |
| Recall (R) | TP / (TP + FN) |
| F1 | 2PR / (P + R) |
| mAP50 | rata-rata AP pada IoU 0,50 |
| mAP50-95 | rata-rata AP pada IoU 0,50; 0,55; ...; 0,95 |
| Akurasi tingkat gambar | gambar dianggap benar jika tidak ada kelas GT yang terlewat dan tidak ada kelas prediksi yang nyasar |
| Presisi/recall/F1 mikro | dihitung dari total TP/FP/FN seluruh kelas |
| F1 makro | rata-rata F1 tiap kelas tanpa pembobotan |

Catatan penting: indikator **tingkat gambar** hanya memeriksa kehadiran kelas,
bukan jumlah instance, IoU, atau kualitas mask. Karena itu akurasi tingkat gambar
bisa lebih tinggi daripada mAP instance. Metrik utama tetap **mask mAP50-95**.

---

## 4. Hasil tahap 6

### 4.1 Skor test per run

| Run | mask P | mask R | mask F1 | mask mAP50 | mask mAP50-95 | box mAP50-95 |
|---|---|---|---|---|---|---|
| yolo11n-seg · s43 | 0.5619 | 0.4443 | 0.4962 | 0.4472 | 0.2468 | 0.3100 |
| yolo26n-seg · s43 | 0.6332 | 0.5385 | 0.5820 | 0.5067 | 0.3085 | 0.3329 |
| yolo11n-seg · s44 | 0.5203 | 0.3921 | 0.4472 | 0.3476 | 0.1950 | 0.2545 |
| yolo26n-seg · s44 | 0.6600 | 0.4650 | 0.5456 | 0.4614 | 0.2441 | 0.3389 |
| yolo11n-seg · s46 | 0.6438 | 0.4771 | 0.5480 | 0.5083 | 0.2746 | 0.3748 |
| yolo26n-seg · s46 | 0.7457 | 0.5778 | 0.6511 | 0.5960 | 0.3310 | 0.4202 |

### 4.2 Rata-rata ± simpangan baku lintas 3 seed (rasio 80)

| Model | mask mAP50-95 | mask mAP50 | mask P | mask R | mask F1 |
|---|---|---|---|---|---|
| yolo11n-seg | 0.2388 ± 0.0330 | 0.4343 ± 0.0662 | 0.5754 | 0.4378 | 0.4972 |
| **yolo26n-seg** | **0.2945 ± 0.0368** | **0.5214 ± 0.0559** | **0.6796** | **0.5271** | **0.5929** |
| Selisih | **+0.0557** | +0.0870 | +0.1043 | +0.0893 | +0.0957 |

| Model | box mAP50-95 | box mAP50 | box P | box R | box F1 |
|---|---|---|---|---|---|
| yolo11n-seg | 0.3131 ± 0.0491 | 0.5230 ± 0.0648 | 0.6604 | 0.4918 | 0.5634 |
| **yolo26n-seg** | **0.3640 ± 0.0398** | **0.6087 ± 0.0608** | **0.7415** | **0.5828** | **0.6517** |
| Selisih | +0.0509 | +0.0857 | +0.0811 | +0.0910 | +0.0883 |

Keunggulan relatif mask mAP50-95: **+23,3%** (0.2945 vs 0.2388).
Selisih berpasangan per split: +0.06170 (s43), +0.04912 (s44), +0.05635 (s46);
rata-rata **+0.05572 ± 0.00515**. `yolo26n-seg` unggul di **3/3 split**.

### 4.3 Skor val (cek generalisasi)

| Model | val mask mAP50-95 | val mask mAP50 | val box mAP50-95 | selisih val − test |
|---|---|---|---|---|
| yolo11n-seg | 0.2622 ± 0.0218 | 0.5246 ± 0.0369 | 0.3502 ± 0.0123 | +0.0234 |
| yolo26n-seg | 0.3306 ± 0.0094 | 0.5881 ± 0.0195 | 0.3994 ± 0.0347 | +0.0361 |

Gap val–test < 0.04, sehingga skor test dapat dipercaya dan tidak ada indikasi
overfitting berarti.

### 4.4 Akurasi per kelas (test mask, mAP50-95)

| Kelas | yolo11n-seg | yolo26n-seg | Selisih |
|---|---:|---:|---:|
| pohon | 0.19256 | 0.23555 | +0.0430 |
| lubang | 0.09217 | 0.11252 | +0.0204 |
| orang | 0.13926 | 0.16153 | +0.0223 |
| motor | 0.38936 | 0.50385 | +0.1145 |
| mobil | 0.38057 | 0.45908 | +0.0785 |

Rincian per seed ada di [`README.md`](../README.md#65-akurasi-per-kelas-test-mask)
dan CSV mentah [`hasil_06/per_kelas.csv`](../hasil_06/per_kelas.csv).

### 4.5 Efisiensi tahap 6

| Model | Parameter | Ukuran bobot | Waktu latih (rata2/run) | Inferensi test |
|---|---|---|---|---|
| yolo11n-seg | 2.859 juta | 5.99 MB | 46.62 menit | 14.88 ms |
| yolo26n-seg | 3.085 juta (+7.9%) | 6.57 MB (+9.7%) | 86.28 menit (+85%) | 14.56 ms |

Latih ~2× lebih lama, tetapi kecepatan inferensi hampir sama.

### 4.6 Kesimpulan tahap 6

1. `yolo26n-seg` lebih akurat pada semua metrik mask dan kotak.
2. Keunggulan konsisten di 3/3 split.
3. Unggul di semua 5 kelas, terbesar pada motor.
4. Biaya utama ada di waktu latih, bukan inferensi.
5. `lubang` masih lemah (< 0.12 mAP50-95) dan menjadi kandidat perbaikan.

---

## 5. Hasil tahap 7

### 5.1 Pilihan final

| Item | Nilai |
|---|---|
| Model | `yolo26n-seg` |
| Rasio | 80 |
| Seed split | 44 (`split_r80_s44`) |
| Alasan | model/rasio dari hasil tahap 6; seed dipilih yang skornya paling dekat median |
| Train final | 3.405 gambar |
| Val-internal | 425 gambar |
| Test asli | 430 gambar |

### 5.2 Konfigurasi latih

| Parameter | Nilai |
|---|---|
| Bobot awal | `yolo26n-seg.pt` (COCO) |
| Epoch | 100 |
| Patience | 30 |
| `imgsz` / `batch` | 416 / 1 |
| Seed latih | 0, `deterministic=True` |
| Optimizer | `auto` |
| `lr0` / `lrf` | 0.01 / 0.01 |
| Momentum / weight decay | 0.937 / 0.0005 |
| Warmup | 3 epoch, momentum 0.8, bias lr 0.1 |
| Mosaic / close_mosaic | 1.0 / 10 |
| HSV h/s/v | 0.015 / 0.7 / 0.4 |
| Translate / scale / fliplr | 0.1 / 0.5 / 0.5 |
| Erasing | 0.4 |
| AMP | true |
| Workers | 0 |
| Waktu latih | **42.87 menit** |
| Parameter | **3.085 juta** |
| Ukuran bobot | **6.57 MB** |

### 5.3 Akurasi test total (430 gambar)

| Metrik | Mask | Kotak |
|---|---:|---:|
| Precision | 0.6774 | 0.7883 |
| Recall | 0.5163 | 0.5951 |
| F1 | 0.5860 | 0.6782 |
| mAP50 | 0.5122 | 0.6709 |
| mAP50-95 | **0.2878** | **0.4028** |

### 5.4 Akurasi per kelas (mask)

| Kelas | n gambar | n instance | P | R | F1 | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| pohon | 95 | 195 | 0.6063 | 0.4028 | 0.4840 | 0.4253 | 0.2841 |
| lubang | 80 | 230 | 0.7835 | 0.3783 | 0.5102 | 0.4840 | 0.2186 |
| orang | 80 | 155 | 0.5986 | 0.6832 | 0.6381 | 0.5069 | 0.2123 |
| motor | 75 | 75 | 0.6820 | 0.2574 | 0.3738 | 0.2874 | 0.1052 |
| mobil | 100 | 250 | 0.7166 | 0.8600 | 0.7818 | 0.8572 | 0.6186 |

### 5.5 Akurasi per kelas (kotak)

| Kelas | n gambar | n instance | P | R | F1 | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| pohon | 95 | 195 | 0.7298 | 0.4848 | 0.5826 | 0.5777 | 0.3445 |
| lubang | 80 | 230 | 0.7294 | 0.3522 | 0.4750 | 0.4661 | 0.2463 |
| orang | 80 | 155 | 0.7908 | 0.9026 | 0.8430 | 0.8984 | 0.4505 |
| motor | 75 | 75 | 0.9647 | 0.3641 | 0.5287 | 0.5400 | 0.2814 |
| mobil | 100 | 250 | 0.7266 | 0.8720 | 0.7927 | 0.8725 | 0.6914 |

### 5.6 Indikator tingkat gambar per kelas (conf ≥ 0.25)

| Kelas | TP | FP | FN | TN | Akurasi | Presisi | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pohon | 95 | 2 | 0 | 333 | 0.9953 | 0.9794 | 1.0000 | 0.9896 |
| lubang | 75 | 1 | 5 | 349 | 0.9860 | 0.9868 | 0.9375 | 0.9615 |
| orang | 80 | 0 | 0 | 350 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| motor | 28 | 0 | 47 | 355 | 0.8907 | 1.0000 | 0.3733 | 0.5437 |
| mobil | 100 | 42 | 0 | 288 | 0.9023 | 0.7042 | 1.0000 | 0.8264 |

Indikator di atas dihitung per kelas dengan pasangan TP/FP/FN/TN pada tingkat
gambar. Akurasi keseluruhan tingkat gambar dilaporkan di tahap 8.

### 5.7 Kecepatan

| Indikator | Nilai |
|---|---|
| Preprocess | 1.5 ms/gambar |
| Inferensi | 19.3 ms/gambar |
| Postprocess | 3.1 ms/gambar |
| Total (r.speed) | ~23.9 ms/gambar |
| Parameter | 3.085 juta |
| Ukuran bobot | 6.57 MB |

Diukur di perangkat latih/evaluasi ini (GTX 1650 Max-Q), bukan Raspberry Pi.

### 5.8 Artefak tahap 7

- [`hasil_07/laporan_final.json`](../hasil_07/laporan_final.json) — seluruh skor test.
- [`hasil_07/laporan_akurasi.txt`](../hasil_07/laporan_akurasi.txt) — ringkasan teks.
- [`hasil_07/per_kelas_test.csv`](../hasil_07/per_kelas_test.csv) — metrik per kelas.
- [`hasil_07/konfusi_tingkat_gambar.csv`](../hasil_07/konfusi_tingkat_gambar.csv) — TP/FP/FN/TN.
- [`hasil_07/grafik/`](../hasil_07/grafik) — grafik total & per kelas.
- [`hasil_07/evaluasi_test/`](../hasil_07/evaluasi_test) — CM & kurva PR/F1/P/R.
- [`hasil_07/runs/`](../hasil_07/runs) — `args.yaml`, `results.png`, plot latih, `weights/`.
- [`models/best.pt`](../models/best.pt) — bobot final.

---

## 6. Hasil tahap 8

### 6.1 Ringkasan test

Model: `models/best.pt` | Split: `split_r80_s44` | Gambar test: 430.

| Metrik | Mask | Kotak |
|---|---:|---:|
| Precision | 0.6774 | 0.7883 |
| Recall | 0.5163 | 0.5951 |
| F1 | 0.5860 | 0.6782 |
| mAP50 | 0.5122 | 0.6709 |
| mAP50-95 | **0.2878** | **0.4028** |

**Tingkat gambar (conf ≥ 0.25):**

| Indikator | Nilai |
|---|---:|
| Gambar benar | 375 / 430 |
| Gambar salah | 55 |
| Akurasi | **0.8721 (87.21%)** |
| Presisi mikro | 0.8936 |
| Recall mikro | 0.8791 |
| F1 mikro | 0.8863 |
| F1 makro | 0.8643 |

### 6.2 Per kelas (mask)

| Kelas | n gambar | n instance | P | R | F1 | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| pohon | 95 | 195 | 0.6063 | 0.4028 | 0.4840 | 0.4253 | 0.2841 |
| lubang | 80 | 230 | 0.7835 | 0.3783 | 0.5102 | 0.4840 | 0.2186 |
| orang | 80 | 155 | 0.5986 | 0.6832 | 0.6381 | 0.5069 | 0.2123 |
| motor | 75 | 75 | 0.6820 | 0.2574 | 0.3738 | 0.2874 | 0.1052 |
| mobil | 100 | 250 | 0.7166 | 0.8600 | 0.7818 | 0.8572 | 0.6186 |

### 6.3 Per kelas (kotak)

| Kelas | n gambar | n instance | P | R | F1 | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| pohon | 95 | 195 | 0.7298 | 0.4848 | 0.5826 | 0.5777 | 0.3445 |
| lubang | 80 | 230 | 0.7294 | 0.3522 | 0.4750 | 0.4661 | 0.2463 |
| orang | 80 | 155 | 0.7908 | 0.9026 | 0.8430 | 0.8984 | 0.4505 |
| motor | 75 | 75 | 0.9647 | 0.3641 | 0.5287 | 0.5400 | 0.2814 |
| mobil | 100 | 250 | 0.7266 | 0.8720 | 0.7927 | 0.8725 | 0.6914 |

### 6.4 Per varian gambar

`ASLI` = subset gambar asli (86); `SEMUA_VARIAN` = seluruh gambar varian non-asli
(344). Baris `asli` di CSV sama dengan `ASLI`.

| Varian | n | mask P | mask R | mask F1 | mask mAP50 | mask mAP50-95 | box P | box R | box F1 | box mAP50 | box mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ASLI | 86 | 0.6707 | 0.5210 | 0.5865 | 0.5257 | **0.3063** | 0.8127 | 0.6086 | 0.6960 | 0.6735 | **0.4252** |
| SEMUA_VARIAN | 344 | 0.6788 | 0.5139 | 0.5849 | 0.5127 | 0.2860 | 0.7813 | 0.5908 | 0.6728 | 0.6727 | 0.4001 |
| gerak_diagonal | 86 | 0.7103 | 0.5022 | 0.5884 | 0.5290 | 0.2991 | 0.8330 | 0.5748 | 0.6802 | 0.6793 | 0.4034 |
| gerak_horizontal | 86 | — | — | — | — | — | — | — | — | — | — |
| gerak_vertikal | 86 | 0.6303 | 0.5499 | 0.5873 | 0.5375 | 0.2953 | 0.7103 | 0.6223 | 0.6634 | 0.6883 | 0.4221 |
| getar | 86 | 0.6837 | 0.5294 | 0.5967 | 0.5187 | 0.2942 | 0.7981 | 0.6121 | 0.6928 | 0.6653 | 0.4036 |

> `gerak_horizontal` gagal dievaluasi per varian karena satu file sementara tidak
> ditemukan (`lubang_171__gerak_horizontal.jpg`). Evaluasi total 430 gambar tetap
> valid; hanya pemecahan per varian tersebut yang tidak tersedia.

### 6.5 Tingkat gambar per kelas

| Kelas | TP | FP | FN | Presisi | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| pohon | 95 | 2 | 0 | 0.9794 | 1.0000 | 0.9896 |
| lubang | 75 | 1 | 5 | 0.9868 | 0.9375 | 0.9615 |
| orang | 80 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| motor | 28 | 0 | 47 | 1.0000 | 0.3733 | 0.5437 |
| mobil | 100 | 42 | 0 | 0.7042 | 1.0000 | 0.8264 |

### 6.6 Analisis kesalahan

Total 55 gambar salah. Pola kesalahan dari
[`hasil_08/kesalahan_gambar.csv`](../hasil_08/kesalahan_gambar.csv):

| Pola | Jumlah | Keterangan |
|---|---:|---|
| motor → mobil | **40** | motor terlewat dan mobil muncul sebagai kelas nyasar |
| lubang terlewat | 5 | lubang tidak terdeteksi |
| motor terlewat | 5 | motor tidak terdeteksi, tanpa kelas nyasar |
| motor → pohon | 2 | motor terlewat, pohon muncul |
| orang + mobil nyasar | 2 | orang terdeteksi, mobil ikut muncul |
| mobil + lubang nyasar | 1 | mobil terdeteksi, lubang ikut muncul |

Kesalahan per varian: asli 13, gerak_diagonal 10, gerak_horizontal 12,
gerak_vertikal 9, getar 11.

Kesimpulan analisis: masalah terbesar adalah **motor mirip mobil** pada sudut/
skala tertentu. Kelas motor punya 75 gambar/75 instance, dan 47 gambar motor
terlewat pada tingkat gambar (recall 0.3733), meskipun presisi tingkat gambar 1.0.

### 6.7 Kecepatan

| Indikator | Nilai |
|---|---:|
| Rata-rata | **21.01 ms/gambar** |
| Median | 19.31 ms |
| Maksimum | 37.54 ms |
| FPS rata-rata | **47.6** |
| Jumlah gambar uji | 100 |
| Preprocess (r.speed) | 1.70 ms |
| Inference (r.speed) | 24.48 ms |
| Postprocess (r.speed) | 3.69 ms |

Diukur di komputer ini (GTX 1650 Max-Q), **bukan** Raspberry Pi. Ukur ulang di Pi
setelah export NCNN.

### 6.8 Artefak tahap 8

- [`hasil_08/ringkasan_test.json`](../hasil_08/ringkasan_test.json) — total, kesalahan, kecepatan.
- [`hasil_08/laporan_akurasi.txt`](../hasil_08/laporan_akurasi.txt) — laporan teks.
- [`hasil_08/per_kelas.csv`](../hasil_08/per_kelas.csv) — metrik per kelas.
- [`hasil_08/per_varian.csv`](../hasil_08/per_varian.csv) — metrik per varian.
- [`hasil_08/tingkat_gambar_per_kelas.csv`](../hasil_08/tingkat_gambar_per_kelas.csv) — TP/FP/FN.
- [`hasil_08/kesalahan_gambar.csv`](../hasil_08/kesalahan_gambar.csv) — 55 gambar salah.
- [`hasil_08/kecepatan.json`](../hasil_08/kecepatan.json) — statistik kecepatan.
- [`hasil_08/grafik/`](../hasil_08/grafik) — 4 grafik ringkasan.
- [`hasil_08/evaluasi_val/`](../hasil_08/evaluasi_val) — CM & kurva PR/F1/P/R.
- [`hasil_08/contoh_prediksi/`](../hasil_08/contoh_prediksi) — 20 benar + 20 salah.

---

## 7. Diskusi

### 7.1 Keunggulan yolo26n-seg

Pada tahap 6, `yolo26n-seg` unggul konsisten di semua metrik dan semua split.
Kenaikan mask mAP50-95 dari 0.2388 ke 0.2945 setara **+23,3% relatif**. Karena
selisih antar-seed kecil (0.0052) dibanding selisih model (0.0557), hasil ini
cukup meyakinkan secara arah, meskipun belum diuji statistik formal.

### 7.2 Hubungan tahap 6 dan tahap 7

Tahap 7 memakai `split_r80_s44`, yaitu split yang pada tahap 6 memberi skor
`yolo26n-seg` paling rendah (0.2441). Setelah latih ulang 100 epoch pada
train+val, skor test naik menjadi **0.2878**. Ini bukan perbandingan apple-to-apple
dengan tahap 6 karena data latih dan jumlah epoch berbeda, tetapi menunjukkan
pelatihan final yang lebih panjang membantu.

### 7.3 Kelas lemah

- **Motor**: mask mAP50-95 0.1052; recall tingkat gambar 0.3733; 40 dari 55
  kesalahan adalah motor yang diprediksi sebagai mobil.
- **Lubang**: mask mAP50-95 0.2186; 5 gambar lubang terlewat. Objek tipis/kecil
  dan variasi permukaan jalan membuat mask sulit.
- **Pohon**: recall mask 0.4028, tetapi pada tingkat gambar recall 1.0; artinya
  pohon biasanya terdeteksi, tetapi kualitas mask/instance-nya rendah.

### 7.4 Pengaruh varian

| Kelompok | mask mAP50-95 | box mAP50-95 |
|---|---:|---:|
| ASLI | 0.3063 | 0.4252 |
| SEMUA_VARIAN | 0.2860 | 0.4001 |
| gerak_diagonal | 0.2991 | 0.4034 |
| gerak_vertikal | 0.2953 | 0.4221 |
| getar | 0.2942 | 0.4036 |

Gambar asli sedikit lebih mudah (0.3063) dibanding gabungan varian (0.2860),
tetapi penurunannya tidak besar. Augmentasi gerak/getar tampaknya cukup membantu
ketahanan model, meskipun evaluasi per varian `gerak_horizontal` tidak tersedia.

### 7.5 Kecepatan

Tahap 6 melaporkan inferensi ~14.5–14.9 ms dari `r.speed` Ultralytics. Tahap 7
melaporkan 19.3 ms inferensi, sedangkan tahap 8 melaporkan 21.01 ms rata-rata
end-to-end untuk 100 gambar. Pada run validasi terpisah, `r.speed` tahap 8
mencatat 24.48 ms inferensi (preprocess 1.70 ms, postprocess 3.69 ms). Perbedaan
ini wajar karena kondisi pengukuran, beban GPU/thermal, dan cakupan waktu
(pre/pasca-proses vs inferensi murni) berbeda. Yang dilaporkan sebagai kecepatan
utama tahap 8 adalah **21.01 ms/gambar** dari pengukuran 100 gambar.

Angka ini bukan angka Raspberry Pi.

---

## 8. Keterbatasan

1. Tahap 6 baru memakai **3 seed split**; belum ada uji signifikansi formal.
2. Tahap 7/8 hanya memakai **satu split final** (`split_r80_s44`).
3. Kecepatan diukur di GTX 1650 Max-Q, bukan perangkat target Raspberry Pi.
4. Evaluasi per varian `gerak_horizontal` gagal karena file sementara hilang.
5. Metrik tingkat gambar hanya memeriksa kehadiran kelas, bukan kualitas instance.
6. Kelas motor dan lubang masih lemah dan perlu data/augmentasi tambahan.
7. `results.csv` log latih tahap 7 tidak tersedia di repo; yang tersedia adalah
   `args.yaml`, `results.png`, dan kurva evaluasi. Tidak ada angka yang
   direkonstruksi secara manual.
8. Folder dataset dan arsip besar tidak di-commit, sehingga reproduksi penuh
   memerlukan dataset lokal.

---

## 9. Reproduksi

### 9.1 Lingkungan

| Komponen | Versi |
|---|---|
| Python | 3.12.10 |
| Ultralytics | 8.4.195 |
| PyTorch | 2.14.1+cu126 |
| Torchvision | 0.29.1+cu126 |
| NumPy | 2.5.4 |
| Matplotlib | 3.11.2 |
| OpenCV | 5.0.0.93 |
| PyYAML | 6.0.3 |
| Pillow | 13.0.0 |
| GPU | NVIDIA GeForce GTX 1650 with Max-Q Design |

Dependensi: [`requirements.txt`](../requirements.txt).

### 9.2 Perintah

```bash
pip install -r requirements.txt

# Tahap 6
python 06_bandingkan_model.py --splits-dir <folder_split> --out hasil_06 \
       --epochs 10 --imgsz 416 --batch 1

# Tahap 7
python 07_latih_final.py --splits-dir <folder_split> --hasil06 hasil_06 \
       --out hasil_07 --models-dir models --epochs 100 --imgsz 416 --batch 1

# Tahap 8
python 08_evaluasi_final.py --splits-dir <folder_split> --hasil07 hasil_07 \
       --model models/best.pt --out hasil_08 --imgsz 416 --batch 1 --conf 0.25
```

### 9.3 Hash artefak

| File | Ukuran | SHA256 |
|---|---:|---|
| `models/best.pt` | 6.567.965 B | `243430f929637ed39f929bc4afc3661ff79b2a777fa917299f38da943692b20a` |
| `hasil_07/runs/final_yolo26n-seg_r80_s44/weights/best.pt` | 6.567.965 B | `243430f929637ed39f929bc4afc3661ff79b2a777fa917299f38da943692b20a` |
| `yolo26n-seg.pt` | 6.719.965 B | `361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f` |
| `yolo11n-seg.pt` | 6.182.636 B | `55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152` |

`models/best.pt` identik byte-per-byte dengan bobot terbaik di `hasil_07/runs/`.

---

## 10. Fuzzy Sugeno (usulan tahap 9)

Skrip [`fuzzysugeno.py`](../fuzzysugeno.py) dan [`jalankan_fuzzy.py`](../jalankan_fuzzy.py)
mengubah keluaran YOLO-seg menjadi tingkat urgensi bahaya 0–100.

**Masukan fuzzy per objek:**

1. Jarak (meter): sangat_dekat / dekat / sedang / jauh.
2. Bahaya kelas (0–10): `lubang`, `motor`, `mobil` = 10; `orang`, `pohon` = 6.
3. Keterhalangan jalur: porsi mask di koridor tengah-bawah gambar (luar / sebagian / penuh).

**Keluaran:** Sugeno orde-nol, rata-rata berbobot 21 aturan.
Level: 0–24 AMAN, 25–49 INFO, 50–74 WASPADA, 75–100 BAHAYA.

`jalankan_fuzzy.py` versi terbaru menambahkan estimasi jarak monokular dari tinggi
bounding box, tinggi referensi kelas, dan FOV vertikal kamera. Estimasi ini harus
dikalibrasi sebelum dipakai untuk keputusan nyata.

Uji cepat `fuzzysugeno.py` (bukan evaluasi akurasi):

| Kelas | Jarak (m) | Jalur | Urgensi | Level |
|---|---:|---:|---:|---|
| pohon | 2.0 | 0.90 | 65.0 | WASPADA |
| lubang | 2.0 | 0.80 | 90.0 | BAHAYA |
| orang | 4.5 | 0.90 | 15.0 | AMAN |
| motor | 3.5 | 0.90 | 60.0 | WASPADA |
| mobil | 3.5 | 0.90 | 60.0 | WASPADA |

Tahap ini **belum punya metrik akurasi** karena belum ada ground truth urgensi.
Jangan mengklaim akurasi fuzzy sampai ada validasi pengguna/sensor.

---

## 11. Kesimpulan

1. `yolo26n-seg` lebih akurat daripada `yolo11n-seg` pada semua metrik dan semua
   seed split; keunggulan mask mAP50-95 +0.0557 (+23,3% relatif).
2. Model final `models/best.pt` mencapai mask mAP50-95 **0.2878**, box mAP50-95
   **0.4028**, dan akurasi tingkat gambar **87,21%** pada 430 gambar test.
3. Kecepatan end-to-end **21,01 ms/gambar (~47,6 FPS)** di GTX 1650 Max-Q; perlu
   diukur ulang di Raspberry Pi.
4. Kelemahan utama: motor sering diprediksi sebagai mobil (40/55 kesalahan) dan
   lubang masih sulit disegmentasi.
5. Langkah berikutnya: tambah data motor/lubang, augmentasi khusus, evaluasi
   per varian yang gagal diulang, dan validasi fuzzy Sugeno dengan ground truth.

---

## 12. Indeks artefak

| Lokasi | Isi |
|---|---|
| [`06_bandingkan_model.py`](../06_bandingkan_model.py) | skrip tahap 6 |
| [`07_latih_final.py`](../07_latih_final.py) | skrip tahap 7 |
| [`08_evaluasi_final.py`](../08_evaluasi_final.py) | skrip tahap 8 |
| [`fuzzysugeno.py`](../fuzzysugeno.py) | aturan fuzzy Sugeno |
| [`jalankan_fuzzy.py`](../jalankan_fuzzy.py) | demo YOLO + fuzzy |
| [`hasil_06/`](../hasil_06) | CSV/JSON/grafik tahap 6 |
| [`hasil_07/`](../hasil_07) | laporan, CSV/JSON, grafik, runs tahap 7 |
| [`hasil_08/`](../hasil_08) | laporan, CSV/JSON, grafik, contoh tahap 8 |
| [`models/best.pt`](../models/best.pt) | bobot final |
| [`docs/DATASET.md`](DATASET.md) | komposisi dataset |
| [`docs/GALERI.md`](GALERI.md) | galeri seluruh grafik & contoh |
| [`requirements.txt`](../requirements.txt) | dependensi |
