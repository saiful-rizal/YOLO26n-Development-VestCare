# Dataset — VestCare

Dokumen ini merangkum komposisi dataset yang dipakai pada tahap 6–8. Dataset
mentah **tidak di-commit** ke GitHub karena ukurannya besar; yang masuk repo
hanya konfigurasi kecil dan statistik turunan.

## 1. Split yang dipakai

| Item | Nilai |
|---|---|
| Dataset | `dataset_colab-001/dataset/split_r80_s44` |
| Rasio split | 80 |
| Seed split | 44 |
| Kelas | pohon, lubang, orang, motor, mobil |
| Jumlah kelas | 5 |

Konfigurasi YAML asli ada di `dataset_colab-001/dataset/split_r80_s44/data.yaml`
(tidak di-commit). Salinan konfigurasi hasil tahap 7:
[`hasil_07/dataset/data.yaml`](../hasil_07/dataset/data.yaml).

## 2. Jumlah gambar dan instance per split

Sumber: [`statistik_split_r80_s44.csv`](statistik_split_r80_s44.csv).

| Split | Gambar | Instance | pohon (inst) | lubang (inst) | orang (inst) | motor (inst) | mobil (inst) |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 3.060 | 6.465 | 1.220 | 1.105 | 1.055 | 1.275 | 1.810 |
| val | 770 | 1.440 | 330 | 280 | 245 | 135 | 450 |
| test | 430 | 905 | 195 | 230 | 155 | 75 | 250 |
| **Total** | **4.260** | **8.810** | **1.745** | **1.615** | **1.455** | **1.485** | **2.510** |

Jumlah gambar per kelas pada test (setiap gambar hanya punya satu kelas, tetapi
bisa memuat beberapa instance):

| Kelas | Gambar test | Instance test |
|---|---:|---:|
| pohon | 95 | 195 |
| lubang | 80 | 230 |
| orang | 80 | 155 |
| motor | 75 | 75 |
| mobil | 100 | 250 |
| **Total** | **430** | **905** |

## 3. Varian gambar

Dataset diperkaya dengan lima varian. Nama file memakai pola
`<sumber>__<varian>.jpg`; file tanpa akhiran varian dianggap `asli`.

Sumber: [`statistik_varian_r80_s44.csv`](statistik_varian_r80_s44.csv).

| Varian | train gambar | train instance | val gambar | val instance | test gambar | test instance |
|---|---:|---:|---:|---:|---:|---:|
| asli | 612 | 1.293 | 154 | 288 | 86 | 181 |
| gerak_diagonal | 612 | 1.293 | 154 | 288 | 86 | 181 |
| gerak_horizontal | 612 | 1.293 | 154 | 288 | 86 | 181 |
| gerak_vertikal | 612 | 1.293 | 154 | 288 | 86 | 181 |
| getar | 612 | 1.293 | 154 | 288 | 86 | 181 |
| **Total** | **3.060** | **6.465** | **770** | **1.440** | **430** | **905** |

Dengan demikian test set seimbang per varian: **86 gambar per varian**, total 430.

## 4. Data latih final tahap 7

Tahap 7 menggabungkan train + val dari `split_r80_s44`, lalu menyisihkan
val-internal 10% secara berkelompok. Angka resmi yang tercatat di
[`hasil_07/laporan_final.json`](../hasil_07/laporan_final.json):

| Bagian | Gambar |
|---|---:|
| Train final | 3.405 |
| Val-internal | 425 |
| Test asli (tidak dipakai latih/val) | 430 |

Val-internal dibentuk per kelompok: semua varian dari gambar sumber yang sama
dan frame berurutan tetap berada di sisi yang sama, sehingga tidak ada kebocoran
antara train final dan val-internal.

## 5. Catatan

- Folder `dataset_colab-001/`, `hasil_07/dataset/images/`, `hasil_07/dataset/labels/`,
  dan `hasil_07/_arsip/` diabaikan oleh Git karena berukuran besar.
- File `hasil_07/dataset/data.yaml` ikut di-commit sebagai dokumentasi konfigurasi;
  jalur `test` di dalamnya menunjuk ke split test asli.
- Statistik di halaman ini dihitung langsung dari file label pada split
  `split_r80_s44`, bukan dari model.
