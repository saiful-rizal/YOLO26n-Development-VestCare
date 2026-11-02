"""TAHAP 9 (usulan) - FUZZY SUGENO: urgensi bahaya benda di depan tunanetra dari keluaran YOLO-seg.

Masukan fuzzy (per benda):
  1. jarak (m)            : sangat_dekat / dekat / sedang / jauh
  2. bahaya kelas (0-10)  : rendah / sedang / tinggi      (tabel BAHAYA_KELAS, atur sesuai kelas model Anda)
  3. keterhalangan jalur  : luar / sebagian / penuh       (porsi mask yang masuk 'koridor jalan' di depan pengguna)
Keluaran Sugeno orde-nol: urgensi 0-100 (konstanta tiap aturan, hasil = rata-rata berbobot).

Level: 0-24 AMAN | 25-49 INFO | 50-74 WASPADA | 75-100 BAHAYA

Contoh pakai dengan YOLO:
    from ultralytics import YOLO
    model = YOLO("models/best.pt")
    res = model.predict(frame, imgsz=640, conf=0.25, verbose=False)[0]
    for d in analisis_hasil(res, jarak_fn=lambda box, nama: 2.0):   # ganti jarak_fn dengan sensor/estimasi
        print(d["pesan"])

Jalankan uji cepat:  python 09_fuzzy_sugeno.py
"""
import numpy as np

# Bobot bahaya per kelas (0-10). SESUAIKAN dengan nama kelas di data.yaml Anda.
BAHAYA_KELAS = {
    "lubang": 10,
    "motor": 10,
    "mobil": 10,
    "orang": 6,
    "pohon": 6,
}
BAHAYA_DEFAULT = 5


# ----------------------------------------------------------------------------- fungsi keanggotaan
def trap(x, a, b, c, d):
    """Trapesium (a<=b<=c<=d). Segitiga bila b==c. Bahu kiri: a==b, bahu kanan: c==d."""
    x = float(x)
    if x < a or x > d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b > a else 1.0
    return (d - x) / (d - c) if d > c else 1.0


def fuzzifikasi(jarak, bahaya, jalur):
    return {
        "jarak": {"sangat_dekat": trap(jarak, -1, 0, 0.8, 1.4),
                  "dekat": trap(jarak, 0.8, 1.4, 2.2, 3.0),
                  "sedang": trap(jarak, 2.2, 3.0, 3.5, 4.5),
                  "jauh": trap(jarak, 3.5, 4.5, 99, 100)},
        "bahaya": {"rendah": trap(bahaya, -1, 0, 3, 5),
                   "sedang": trap(bahaya, 3, 5, 6, 8),
                   "tinggi": trap(bahaya, 6, 8, 10, 11)},
        "jalur": {"luar": trap(jalur, -1, 0, 0.1, 0.35),
                  "sebagian": trap(jalur, 0.1, 0.35, 0.5, 0.75),
                  "penuh": trap(jalur, 0.5, 0.75, 1, 2)},
    }


# ----------------------------------------------------------------------------- basis aturan Sugeno
# (jarak, bahaya, jalur, urgensi)  ; None = bebas
ATURAN = [
    ("sangat_dekat", "tinggi", "penuh", 100), ("sangat_dekat", "tinggi", "sebagian", 100),
    ("dekat", "tinggi", "penuh", 90), ("dekat", "tinggi", "sebagian", 75),
    ("sedang", "tinggi", "penuh", 60), ("sedang", "tinggi", "sebagian", 45),
    ("jauh", "tinggi", None, 30),
    ("sangat_dekat", "sedang", "penuh", 85), ("sangat_dekat", "sedang", "sebagian", 65),
    ("dekat", "sedang", "penuh", 65), ("dekat", "sedang", "sebagian", 50),
    ("sedang", "sedang", "penuh", 40), ("sedang", "sedang", "sebagian", 30),
    ("jauh", "sedang", None, 15),
    ("sangat_dekat", "rendah", "penuh", 60), ("sangat_dekat", "rendah", "sebagian", 40),
    ("dekat", "rendah", "penuh", 40), ("dekat", "rendah", "sebagian", 25),
    ("sedang", "rendah", None, 15), ("jauh", "rendah", None, 5),
    (None, None, "luar", 5),   # di luar jalur: hampir selalu aman
]


def sugeno(jarak, bahaya, jalur):
    mu = fuzzifikasi(jarak, bahaya, jalur)
    num = den = 0.0
    for j, b, p, z in ATURAN:
        w = min(mu["jarak"][j] if j else 1.0, mu["bahaya"][b] if b else 1.0, mu["jalur"][p] if p else 1.0)
        num += w * z
        den += w
    return num / den if den > 1e-9 else 0.0


def level(u):
    return "BAHAYA" if u >= 75 else "WASPADA" if u >= 50 else "INFO" if u >= 25 else "AMAN"


# ----------------------------------------------------------------------------- hubungan dengan keluaran YOLO
def keterhalangan(mask_poly, w, h, koridor=(0.30, 0.70), y_atas=0.45):
    """Porsi mask yang berada di koridor tengah-bawah gambar (jalur jalan pengguna). mask_poly: Nx2 piksel."""
    if mask_poly is None or len(mask_poly) < 3:
        return 0.0
    import cv2
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.asarray(mask_poly, np.int32)], 1)
    kor = np.zeros_like(m)
    kor[int(y_atas * h):, int(koridor[0] * w):int(koridor[1] * w)] = 1
    luas = m.sum()
    return float((m & kor).sum() / luas) if luas else 0.0


def arah(cx_norm):
    return "kiri" if cx_norm < 0.4 else "kanan" if cx_norm > 0.6 else "depan"


def analisis_hasil(res, jarak_fn, conf_min=0.35):
    """res: hasil ultralytics. jarak_fn(box_xyxy, nama_kelas)->meter. Return daftar dict diurut dari paling gawat."""
    h, w = res.orig_shape
    out = []
    if res.boxes is None or len(res.boxes) == 0:
        return out
    polys = res.masks.xy if res.masks is not None else [None] * len(res.boxes)
    for box, poly in zip(res.boxes, polys):
        conf = float(box.conf)
        if conf < conf_min:
            continue
        nama = res.names[int(box.cls)]
        xyxy = box.xyxy[0].tolist()
        jarak = jarak_fn(xyxy, nama)
        bahaya = BAHAYA_KELAS.get(nama, BAHAYA_DEFAULT)
        if poly is None:  # tanpa mask: pakai kotak sebagai pengganti
            poly = np.array([[xyxy[0], xyxy[1]], [xyxy[2], xyxy[1]], [xyxy[2], xyxy[3]], [xyxy[0], xyxy[3]]])
        jalur = keterhalangan(poly, w, h)
        u = sugeno(jarak, bahaya, jalur)
        pos = arah((xyxy[0] + xyxy[2]) / 2 / w)
        out.append({"kelas": nama, "jarak_m": jarak, "bahaya": bahaya, "jalur": round(jalur, 2), "conf": round(conf, 2),
                    "urgensi": round(u, 1), "level": level(u), "arah": pos,
                    "pesan": f"{level(u)}: {nama} {pos}, sekitar {jarak:.1f} meter"})
    return sorted(out, key=lambda d: -d["urgensi"])


if __name__ == "__main__":
    uji = [("pohon", 2.0, 0.9), ("lubang", 2.0, 0.8), ("orang", 4.5, 0.9),
           ("motor", 3.5, 0.9), ("mobil", 3.5, 0.9)]
    print(f"{'kelas':<12}{'jarak':>6}{'jalur':>7}{'urgensi':>9}  level")
    for k, j, p in uji:
        u = sugeno(j, BAHAYA_KELAS.get(k, BAHAYA_DEFAULT), p)
        print(f"{k:<12}{j:>6.1f}{p:>7.2f}{u:>9.1f}  {level(u)}")