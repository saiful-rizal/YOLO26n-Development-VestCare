# Galeri Hasil — VestCare YOLO26n-seg

Halaman ini menampilkan seluruh grafik dan contoh prediksi yang tersimpan di repo.
Kembali ke [laporan lengkap](LAPORAN_LENGKAP.md) atau [README](../README.md).

---

## Tahap 6 — Perbandingan model

### Perbandingan model x rasio

![Perbandingan model x rasio](../hasil_06/grafik/perbandingan_model_rasio.png)

### Confusion matrix test, seed s46

| yolo11n-seg | yolo26n-seg |
|---|---|
| ![CM 11n s46](../runs/segment/hasil_06/plots/yolo11n-seg__split_r80_s46_test/confusion_matrix.png) | ![CM 26n s46](../runs/segment/hasil_06/plots/yolo26n-seg__split_r80_s46_test/confusion_matrix.png) |

### Kurva mask dan kotak, seed s46

| Kurva | yolo11n-seg | yolo26n-seg |
|---|---|---|
| MaskPR_curve.png | ![MaskPR 11n](../runs/segment/hasil_06/plots/yolo11n-seg__split_r80_s46_test/MaskPR_curve.png) | ![MaskPR 26n](../runs/segment/hasil_06/plots/yolo26n-seg__split_r80_s46_test/MaskPR_curve.png) |
| MaskF1_curve.png | ![MaskF1 11n](../runs/segment/hasil_06/plots/yolo11n-seg__split_r80_s46_test/MaskF1_curve.png) | ![MaskF1 26n](../runs/segment/hasil_06/plots/yolo26n-seg__split_r80_s46_test/MaskF1_curve.png) |
| MaskP_curve.png | ![MaskP 11n](../runs/segment/hasil_06/plots/yolo11n-seg__split_r80_s46_test/MaskP_curve.png) | ![MaskP 26n](../runs/segment/hasil_06/plots/yolo26n-seg__split_r80_s46_test/MaskP_curve.png) |
| MaskR_curve.png | ![MaskR 11n](../runs/segment/hasil_06/plots/yolo11n-seg__split_r80_s46_test/MaskR_curve.png) | ![MaskR 26n](../runs/segment/hasil_06/plots/yolo26n-seg__split_r80_s46_test/MaskR_curve.png) |
| BoxPR_curve.png | ![BoxPR 11n](../runs/segment/hasil_06/plots/yolo11n-seg__split_r80_s46_test/BoxPR_curve.png) | ![BoxPR 26n](../runs/segment/hasil_06/plots/yolo26n-seg__split_r80_s46_test/BoxPR_curve.png) |
| BoxF1_curve.png | ![BoxF1 11n](../runs/segment/hasil_06/plots/yolo11n-seg__split_r80_s46_test/BoxF1_curve.png) | ![BoxF1 26n](../runs/segment/hasil_06/plots/yolo26n-seg__split_r80_s46_test/BoxF1_curve.png) |
| BoxP_curve.png | ![BoxP 11n](../runs/segment/hasil_06/plots/yolo11n-seg__split_r80_s46_test/BoxP_curve.png) | ![BoxP 26n](../runs/segment/hasil_06/plots/yolo26n-seg__split_r80_s46_test/BoxP_curve.png) |
| BoxR_curve.png | ![BoxR 11n](../runs/segment/hasil_06/plots/yolo11n-seg__split_r80_s46_test/BoxR_curve.png) | ![BoxR 26n](../runs/segment/hasil_06/plots/yolo26n-seg__split_r80_s46_test/BoxR_curve.png) |

### Log latih (results.png), seed s46

| yolo11n-seg | yolo26n-seg |
|---|---|
| ![results 11n](../runs/segment/hasil_06/runs/yolo11n-seg__split_r80_s46/results.png) | ![results 26n](../runs/segment/hasil_06/runs/yolo26n-seg__split_r80_s46/results.png) |

---

## Tahap 7 — Model final `yolo26n-seg` (r80 s44)

### Ringkasan

| Ringkasan total | Per kelas |
|---|---|
| ![Ringkasan total](../hasil_07/grafik/1_ringkasan_total.png) | ![Per kelas](../hasil_07/grafik/2_per_kelas.png) |

### Konfusi tingkat gambar

![Konfusi tingkat gambar](../hasil_07/konfusi_tingkat_gambar.png)

### Evaluasi test asli

| Kurva | Gambar |
|---|---|
| BoxF1_curve.png | ![BoxF1](../hasil_07/evaluasi_test/final_yolo26n-seg_r80_s44/BoxF1_curve.png) |
| BoxP_curve.png | ![BoxP](../hasil_07/evaluasi_test/final_yolo26n-seg_r80_s44/BoxP_curve.png) |
| BoxPR_curve.png | ![BoxPR](../hasil_07/evaluasi_test/final_yolo26n-seg_r80_s44/BoxPR_curve.png) |
| BoxR_curve.png | ![BoxR](../hasil_07/evaluasi_test/final_yolo26n-seg_r80_s44/BoxR_curve.png) |
| confusion_matrix.png | ![Confusion matrix](../hasil_07/evaluasi_test/final_yolo26n-seg_r80_s44/confusion_matrix.png) |
| confusion_matrix_normalized.png | ![Confusion matrix normalized](../hasil_07/evaluasi_test/final_yolo26n-seg_r80_s44/confusion_matrix_normalized.png) |
| MaskF1_curve.png | ![MaskF1](../hasil_07/evaluasi_test/final_yolo26n-seg_r80_s44/MaskF1_curve.png) |
| MaskP_curve.png | ![MaskP](../hasil_07/evaluasi_test/final_yolo26n-seg_r80_s44/MaskP_curve.png) |
| MaskPR_curve.png | ![MaskPR](../hasil_07/evaluasi_test/final_yolo26n-seg_r80_s44/MaskPR_curve.png) |
| MaskR_curve.png | ![MaskR](../hasil_07/evaluasi_test/final_yolo26n-seg_r80_s44/MaskR_curve.png) |

### Log latih

| results.png | labels.jpg |
|---|---|
| ![results](../hasil_07/runs/final_yolo26n-seg_r80_s44/results.png) | ![labels](../hasil_07/runs/final_yolo26n-seg_r80_s44/labels.jpg) |

---

## Tahap 8 — Evaluasi mendalam `models/best.pt`

### Grafik ringkasan

| Ringkasan total | Per kelas |
|---|---|
| ![Ringkasan total](../hasil_08/grafik/1_ringkasan_total.png) | ![Per kelas](../hasil_08/grafik/2_per_kelas.png) |

| Per varian | Tingkat gambar |
|---|---|
| ![Per varian](../hasil_08/grafik/3_per_varian.png) | ![Tingkat gambar](../hasil_08/grafik/4_tingkat_gambar.png) |

### Evaluasi val/test

| Kurva | Gambar |
|---|---|
| BoxF1_curve.png | ![BoxF1](../hasil_08/evaluasi_val/BoxF1_curve.png) |
| BoxP_curve.png | ![BoxP](../hasil_08/evaluasi_val/BoxP_curve.png) |
| BoxPR_curve.png | ![BoxPR](../hasil_08/evaluasi_val/BoxPR_curve.png) |
| BoxR_curve.png | ![BoxR](../hasil_08/evaluasi_val/BoxR_curve.png) |
| confusion_matrix.png | ![Confusion matrix](../hasil_08/evaluasi_val/confusion_matrix.png) |
| confusion_matrix_normalized.png | ![Confusion matrix normalized](../hasil_08/evaluasi_val/confusion_matrix_normalized.png) |
| MaskF1_curve.png | ![MaskF1](../hasil_08/evaluasi_val/MaskF1_curve.png) |
| MaskP_curve.png | ![MaskP](../hasil_08/evaluasi_val/MaskP_curve.png) |
| MaskPR_curve.png | ![MaskPR](../hasil_08/evaluasi_val/MaskPR_curve.png) |
| MaskR_curve.png | ![MaskR](../hasil_08/evaluasi_val/MaskR_curve.png) |

### Contoh prediksi benar (20 gambar)

| Gambar | Gambar |
|---|---|
| ![lubang_raw_164](../hasil_08/contoh_prediksi/benar/lubang_raw_164.jpg) | ![lubang_raw_174__gerak_vertikal](../hasil_08/contoh_prediksi/benar/lubang_raw_174__gerak_vertikal.jpg) |
| ![lubang_raw_174__getar](../hasil_08/contoh_prediksi/benar/lubang_raw_174__getar.jpg) | ![mobil_IMG_4197_0082](../hasil_08/contoh_prediksi/benar/mobil_IMG_4197_0082.jpg) |
| ![mobil_IMG_4199p1_1017__getar](../hasil_08/contoh_prediksi/benar/mobil_IMG_4199p1_1017__getar.jpg) | ![mobil_test_IMG_19674__getar](../hasil_08/contoh_prediksi/benar/mobil_test_IMG_19674__getar.jpg) |
| ![mobil_test_IMG_19756__gerak_horizontal](../hasil_08/contoh_prediksi/benar/mobil_test_IMG_19756__gerak_horizontal.jpg) | ![mobil_video1_f0000014_1](../hasil_08/contoh_prediksi/benar/mobil_video1_f0000014_1.jpg) |
| ![motor_IMG_00767__gerak_vertikal](../hasil_08/contoh_prediksi/benar/motor_IMG_00767__gerak_vertikal.jpg) | ![orang_25](../hasil_08/contoh_prediksi/benar/orang_25.jpg) |
| ![orang_49](../hasil_08/contoh_prediksi/benar/orang_49.jpg) | ![orang_70__gerak_horizontal](../hasil_08/contoh_prediksi/benar/orang_70__gerak_horizontal.jpg) |
| ![pohon_sekunder_pohon-114__gerak_horizontal](../hasil_08/contoh_prediksi/benar/pohon_sekunder_pohon-114__gerak_horizontal.jpg) | ![pohon_sekunder_pohon-123__gerak_diagonal](../hasil_08/contoh_prediksi/benar/pohon_sekunder_pohon-123__gerak_diagonal.jpg) |
| ![pohon_sekunder_pohon-154__gerak_vertikal](../hasil_08/contoh_prediksi/benar/pohon_sekunder_pohon-154__gerak_vertikal.jpg) | ![pohon_sekunder_pohon-162__gerak_horizontal](../hasil_08/contoh_prediksi/benar/pohon_sekunder_pohon-162__gerak_horizontal.jpg) |
| ![pohon_sekunder_pohon-28__gerak_horizontal](../hasil_08/contoh_prediksi/benar/pohon_sekunder_pohon-28__gerak_horizontal.jpg) | ![pohon_sekunder_pohon-67](../hasil_08/contoh_prediksi/benar/pohon_sekunder_pohon-67.jpg) |
| ![pohon_sekunder_pohon-84__gerak_diagonal](../hasil_08/contoh_prediksi/benar/pohon_sekunder_pohon-84__gerak_diagonal.jpg) | ![pohon_sekunder_pohon-89__gerak_vertikal](../hasil_08/contoh_prediksi/benar/pohon_sekunder_pohon-89__gerak_vertikal.jpg) |

### Contoh prediksi salah (20 gambar)

| Gambar | Gambar |
|---|---|
| ![lubang_raw_126__getar](../hasil_08/contoh_prediksi/salah/lubang_raw_126__getar.jpg) | ![motor_IMG_00608](../hasil_08/contoh_prediksi/salah/motor_IMG_00608.jpg) |
| ![motor_IMG_00608__gerak_horizontal](../hasil_08/contoh_prediksi/salah/motor_IMG_00608__gerak_horizontal.jpg) | ![motor_IMG_00771__gerak_diagonal](../hasil_08/contoh_prediksi/salah/motor_IMG_00771__gerak_diagonal.jpg) |
| ![motor_IMG_00771__getar](../hasil_08/contoh_prediksi/salah/motor_IMG_00771__getar.jpg) | ![motor_IMG_00831__gerak_diagonal](../hasil_08/contoh_prediksi/salah/motor_IMG_00831__gerak_diagonal.jpg) |
| ![motor_IMG_00831__gerak_horizontal](../hasil_08/contoh_prediksi/salah/motor_IMG_00831__gerak_horizontal.jpg) | ![motor_IMG_00999](../hasil_08/contoh_prediksi/salah/motor_IMG_00999.jpg) |
| ![motor_IMG_01033__gerak_diagonal](../hasil_08/contoh_prediksi/salah/motor_IMG_01033__gerak_diagonal.jpg) | ![motor_IMG_01033__gerak_horizontal](../hasil_08/contoh_prediksi/salah/motor_IMG_01033__gerak_horizontal.jpg) |
| ![motor_IMG_01033__gerak_vertikal](../hasil_08/contoh_prediksi/salah/motor_IMG_01033__gerak_vertikal.jpg) | ![motor_IMG_01111__gerak_vertikal](../hasil_08/contoh_prediksi/salah/motor_IMG_01111__gerak_vertikal.jpg) |
| ![motor_IMG_01111__getar](../hasil_08/contoh_prediksi/salah/motor_IMG_01111__getar.jpg) | ![motor_IMG_01112__gerak_horizontal](../hasil_08/contoh_prediksi/salah/motor_IMG_01112__gerak_horizontal.jpg) |
| ![motor_IMG_01183__getar](../hasil_08/contoh_prediksi/salah/motor_IMG_01183__getar.jpg) | ![motor_IMG_01275__gerak_diagonal](../hasil_08/contoh_prediksi/salah/motor_IMG_01275__gerak_diagonal.jpg) |
| ![motor_IMG_19230](../hasil_08/contoh_prediksi/salah/motor_IMG_19230.jpg) | ![motor_IMG_19230__gerak_diagonal](../hasil_08/contoh_prediksi/salah/motor_IMG_19230__gerak_diagonal.jpg) |
| ![motor_IMG_19230__getar](../hasil_08/contoh_prediksi/salah/motor_IMG_19230__getar.jpg) | ![orang_25__getar](../hasil_08/contoh_prediksi/salah/orang_25__getar.jpg) |

---

Folder sumber: [`hasil_06/`](../hasil_06), [`hasil_07/`](../hasil_07), [`hasil_08/`](../hasil_08).
