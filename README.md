# TAHAP 6 — Perbandingan `yolo11n-seg` vs `yolo26n-seg` (VestCare)

Eksperimen segmentasi objek jalan untuk penunjang tunanetra (rompi VestCare):
membandingkan **YOLO11n-seg** dan **YOLO26n-seg** pada dataset yang sama, dengan
hiperparameter identik, lalu mengevaluasi bobot terbaik pada split **val** dan **test**.

Semua angka di halaman ini diambil langsung dari file CSV di [`hasil_06/`](hasil_06/).

---

## 1. Pengaturan eksperimen

| Parameter | Nilai |
|---|---|
| Model | `yolo11n-seg.pt`, `yolo26n-seg.pt` (bobot pralatih Ultralytics) |
| Kelas | pohon, lubang, orang, motor, mobil |
| Rasio split train/val/test | 80 (sisa val+test) |
| Seed split | 43, 44, 46 (**3 ulangan** per model) |
| Epoch | 10, `patience` 30 |
| `imgsz` / `batch` | 416 / 1 |
| Seed latih | 0, `deterministic=True` |
| Optimizer | resep bawaan masing-masing model (`auto`) |
| Pemilihan model/rasio | memakai skor **test** (bukan val) |

Run: 2 model × 3 seed split = **6 pelatihan**, masing-masing dievaluasi di val dan test.

---

## 2. Hasil utama per run (skor **test**)

| Run | mask P | mask R | mask F1 | mask mAP50 | **mask mAP50-95** | box mAP50-95 |
|---|---|---|---|---|---|---|
| yolo11n-seg · s43 | 0.5619 | 0.4443 | 0.4962 | 0.4472 | 0.2468 | 0.3100 |
| yolo26n-seg · s43 | 0.6332 | 0.5385 | 0.5820 | 0.5067 | **0.3085** | 0.3329 |
| yolo11n-seg · s44 | 0.5203 | 0.3921 | 0.4472 | 0.3476 | 0.1950 | 0.2545 |
| yolo26n-seg · s44 | 0.6600 | 0.4650 | 0.5456 | 0.4614 | **0.2441** | 0.3389 |
| yolo11n-seg · s46 | 0.6438 | 0.4771 | 0.5480 | 0.5083 | 0.2746 | 0.3748 |
| yolo26n-seg · s46 | 0.7457 | 0.5778 | 0.6511 | 0.5960 | **0.3310** | 0.4202 |

---

## 3. Ringkasan rata-rata simpangan baku lintas 3 seed (rasio 80)

### 3.1 Skor test (akurasi)

| Model | mask mAP50-95 | mask mAP50 | mask P | mask R | mask F1 |
|---|---|---|---|---|---|
| yolo11n-seg | 0.2388 ± 0.0330 | 0.4343 ± 0.0662 | 0.5754 | 0.4378 | 0.4972 |
| **yolo26n-seg** | **0.2945 ± 0.0368** | **0.5214 ± 0.0559** | **0.6796** | **0.5271** | **0.5929** |
| Selisih (26n − 11n) | **+0.0557** | +0.0870 | +0.1043 | +0.0893 | +0.0957 |

| Model | box mAP50-95 | box mAP50 | box P | box R | box F1 |
|---|---|---|---|---|---|
| yolo11n-seg | 0.3131 ± 0.0491 | 0.5230 ± 0.0648 | 0.6604 | 0.4918 | 0.5634 |
| **yolo26n-seg** | **0.3640 ± 0.0398** | **0.6087 ± 0.0608** | **0.7415** | **0.5828** | **0.6517** |
| Selisih (26n − 11n) | +0.0509 | +0.0857 | +0.0811 | +0.0910 | +0.0883 |

### 3.2 Skor val (untuk cek generalisasi)

| Model | val mask mAP50-95 | val mask mAP50 | val box mAP50-95 | selisih val − test |
|---|---|---|---|---|
| yolo11n-seg | 0.2622 ± 0.0218 | 0.5246 ± 0.0369 | 0.3502 ± 0.0123 | +0.0234 |
| yolo26n-seg | 0.3306 ± 0.0094 | 0.5881 ± 0.0195 | 0.3994 ± 0.0347 | +0.0361 |

Gap val–test kecil (< 0.04), sehingga skor test dapat dipercaya dan tidak ada overfitting berarti.

---

## 4. Perbandingan berpasangan per split (test mask mAP50-95)

Selisih dihitung pada split dengan seed yang sama (`ringkasan_model.csv`):

| Split | yolo26n-seg − yolo11n-seg |
|---|---|
| split_r80_s43 | +0.06170 |
| split_r80_s44 | +0.04912 |
| split_r80_s46 | +0.05635 |
| **Rata-rata** | **+0.05572 ± 0.00515** |

`yolo26n-seg` **unggul di 3 dari 3 split** (100%).

> Catatan kejujuran ilmiah: selisih konsisten dan sebanding simpangannya
> (0.0557 vs 0.0052), tetapi baru **n = 3 seed**. Untuk klaim statistik yang kuat,
> tambah seed split di tahap 5 (mis. `--seed 47 48 49`).

---

## 5. Akurasi per kelas (test mask)

### 5.1 `yolo11n-seg` — mAP50-95 per seed

| Kelas | s43 | s44 | s46 | **Rata-rata** | P (rata2) | R (rata2) | mAP50 (rata2) |
|---|---|---|---|---|---|---|---|
| pohon | 0.19594 | 0.19978 | 0.18195 | **0.19256** | 0.622 | 0.309 | 0.319 |
| lubang | 0.03857 | 0.06997 | 0.16796 | **0.09217** | 0.396 | 0.285 | 0.216 |
| orang | 0.11453 | 0.14186 | 0.16139 | **0.13926** | 0.424 | 0.429 | 0.389 |
| motor | 0.52295 | 0.24308 | 0.40204 | **0.38936** | 0.720 | 0.612 | 0.648 |
| mobil | 0.36193 | 0.32013 | 0.45965 | **0.38057** | 0.713 | 0.554 | 0.599 |

### 5.2 `yolo26n-seg` — mAP50-95 per seed

| Kelas | s43 | s44 | s46 | **Rata-rata** | P (rata2) | R (rata2) | mAP50 (rata2) |
|---|---|---|---|---|---|---|---|
| pohon | 0.21351 | 0.25370 | 0.23944 | **0.23555** | 0.767 | 0.368 | 0.401 |
| lubang | 0.03551 | 0.08834 | 0.21372 | **0.11252** | 0.430 | 0.248 | 0.246 |
| orang | 0.12248 | 0.19231 | 0.16981 | **0.16153** | 0.601 | 0.647 | 0.515 |
| motor | 0.75489 | 0.20599 | 0.55068 | **0.50385** | 0.787 | 0.801 | 0.761 |
| mobil | 0.41605 | 0.48008 | 0.48112 | **0.45908** | 0.814 | 0.571 | 0.684 |

### 5.3 Selisih rata-rata per kelas (`yolo26n-seg` − `yolo11n-seg`)

| Kelas | yolo11n-seg | yolo26n-seg | Selisih |
|---|---|---|---|
| pohon | 0.19256 | 0.23555 | +0.0430 |
| lubang | 0.09217 | 0.11252 | +0.0204 |
| orang | 0.13926 | 0.16153 | +0.0223 |
| motor | 0.38936 | 0.50385 | +0.1145 |
| mobil | 0.38057 | 0.45908 | +0.0785 |

Kedua model paling kuat pada **motor** dan **mobil**, paling lemah pada **lubang**
(objek kecil/tipis). `yolo26n-seg` lebih unggul di **semua kelas**, terbesar pada motor.
Data mentah per run: [`hasil_06/per_kelas.csv`](hasil_06/per_kelas.csv).

---

## 6. Grafik

### 6.1 Perbandingan model × rasio split (test mask mAP50-95 ± sd, dan seluruh indikator)

![Perbandingan model x rasio](hasil_06/grafik/perbandingan_model_rasio.png)

### 6.2 Confusion matrix pada test (seed s46)

| yolo11n-seg | yolo26n-seg |
|---|---|
| ![CM 11n](runs/segment/hasil_06/plots/yolo11n-seg__split_r80_s46_test/confusion_matrix.png) | ![CM 26n](runs/segment/hasil_06/plots/yolo26n-seg__split_r80_s46_test/confusion_matrix.png) |

### 6.3 Kurva PR mask pada test (seed s46)

| yolo11n-seg | yolo26n-seg |
|---|---|
| ![PR 11n](runs/segment/hasil_06/plots/yolo11n-seg__split_r80_s46_test/MaskPR_curve.png) | ![PR 26n](runs/segment/hasil_06/plots/yolo26n-seg__split_r80_s46_test/MaskPR_curve.png) |

### 6.4 Kurva F1 mask pada test (seed s46)

| yolo11n-seg | yolo26n-seg |
|---|---|
| ![F1 11n](runs/segment/hasil_06/plots/yolo11n-seg__split_r80_s46_test/MaskF1_curve.png) | ![F1 26n](runs/segment/hasil_06/plots/yolo26n-seg__split_r80_s46_test/MaskF1_curve.png) |

Grafik lain (kurva P/R, box, val, `results.png` latih per run) ada di
[`runs/segment/hasil_06/`](runs/segment/hasil_06).

---

## 7. Efisiensi

| Model | Parameter | Ukuran bobot | Waktu latih (rata2/run) | Inferensi test | 
|---|---|---|---|---|
| yolo11n-seg | 2.859 juta | 5.99 MB | 46.62 menit | 14.88 ms |
| yolo26n-seg | 3.085 juta (+7.9%) | 6.57 MB (+9.7%) | 86.28 menit (+85%) | 14.56 ms |

Kecepatan inferensi **hampir sama** (~14.5–14.9 ms), yolo26n-seg lebih berat saat
latih tetapi tidak lebih lambat saat inferensi.

---

## 8. Kesimpulan

1. **`yolo26n-seg` lebih akurat di semua metrik**: mask mAP50-95 0.2945 vs 0.2388
   (+23.3% relatif), mAP50 0.5214 vs 0.4343, F1 0.5929 vs 0.4972.
2. Keunggulan **konsisten di 3/3 split** dengan selisih rata-rata **+0.0557**.
3. Unggul di **semua 5 kelas**, terbesar pada kelas motor.
4. Biaya: latih ~2× lebih lama, parameter +7.9%, tetapi **inferensi sama cepat**
   layak dipakai untuk deployment di rompi.
5. Kelas `lubang` masih rendah (< 0.12 mAP50-95) kandidat perbaikan
   (augmentasi/annotasi khusus) di tahap selanjutnya.

---

## 9. Struktur repo (hanya tahap 6)

```
06_bandingkan_model.py      skrip perbandingan model x split
hasil_06/
  hasil_run.csv             1 run per (model, split): semua metrik test & val
  per_kelas.csv             mAP/P/R per kelas pada test
  ringkasan_rasio.csv       rata-rata ± sd lintas seed per (model, rasio)
  ringkasan_model.csv       selisih berpasangan yolo26n - yolo11n
  grafik/perbandingan_model_rasio.png
  run_json/*.json           cache per run (run ulang dilewati)
runs/segment/hasil_06/
  runs/<run>/               results.csv/png, kurva, confusion matrix, weights/*.pt
  plots/<run>_{val,test}/   evaluasi val & test per run
yolo11n-seg.pt, yolo26n-seg.pt   bobot pralatih
```

## 10. Menjalankan ulang

```bash
pip install -U ultralytics matplotlib
# dataset split (split_r80_s43, s44, s46) harus sudah ada di --splits-dir
python 06_bandingkan_model.py --splits-dir <folder_split> --out hasil_06 \
       --epochs 10 --imgsz 416 --batch 1
```

Di Colab:

```python
!pip install -U ultralytics
!python 06_bandingkan_model.py --splits-dir /content/dataset --out /content/hasil_06 --epochs 10
```

Run yang sudah punya JSON di `hasil_06/run_json/` otomatis dilewati (resumable).
