"""TAHAP 8 - EVALUASI MENDALAM model final (models/best.pt) pada TEST ASLI.

Yang dihasilkan (folder --out):
  ringkasan_test.json      skor total di test: mAP kotak/mask, P, R, kecepatan bawaan Ultralytics
  per_kelas.csv            P, R, mAP50, mAP50-95 (mask) tiap kelas
  per_varian.csv           skor dipisah per jenis gambar (asli, gerak_*, cahaya, dll. dari nama file
                           '<sumber>__<varian>.jpg') + baris 'ASLI' vs 'SEMUA_VARIAN'
                           -> bukti apakah pengayaan tahap 3 membantu model tahan gerak/cahaya
  kesalahan_gambar.csv     tiap gambar test yang salah tingkat-gambar (ada kelas terlewat / kelas nyasar)
  grafik/                  PNG: total, per kelas, per varian, tingkat-gambar
  laporan_akurasi.txt      ringkasan lengkap (sama dengan yang tercetak di layar)
  tingkat_gambar_per_kelas.csv  TP/FP/FN, P, R, F1 per kelas pada tingkat gambar
  kecepatan.json           waktu inferensi satu-gambar (batch 1) di perangkat ini
  evaluasi_val/            confusion matrix & kurva PR dari Ultralytics
  contoh_prediksi/salah/   gambar hasil prediksi yang salah (bahan analisis laporan)
  contoh_prediksi/benar/   contoh acak yang benar

Catatan: kecepatan di sini adalah kecepatan di komputer yang menjalankan skrip ini (mis. GPU
temanmu), BUKAN di Raspberry Pi. Ukur ulang di Pi setelah export NCNN (tahap 9/10).

Jalankan:
  python scripts/08_evaluasi_final.py --splits-dir <folder split_r*_s*>
Di komputer tanpa config.py (mis. GPU teman), semua path diberikan lewat opsi:
  python 08_evaluasi_final.py --splits-dir D:\\kerja\\dataset --hasil07 D:\\kerja\\hasil_07 \\
         --model D:\\kerja\\models\\best.pt --out D:\\kerja\\hasil_08 --device 0
"""
import argparse
import csv
import json
import random
import shutil
import statistics
import time
from collections import defaultdict
from pathlib import Path

try:
    from config import ANN_DIR, HASIL
except ImportError:  # dijalankan tanpa config
    ANN_DIR = HASIL = None

EKSTENSI = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------- util
def daftar_test(split_dir):
    d = Path(split_dir) / "images" / "test"
    if not d.is_dir():
        raise SystemExit(f"Tidak ada {d}. Split ini tidak punya test set.")
    return sorted(p for p in d.iterdir() if p.suffix.lower() in EKSTENSI)


def nama_varian(stem):
    return stem.split("__", 1)[1] if "__" in stem else "asli"


def kelas_gt(split_dir, stem):
    lbl = Path(split_dir) / "labels" / "test" / f"{stem}.txt"
    if not lbl.exists():
        return set()
    out = set()
    for baris in lbl.read_text().splitlines():
        bagian = baris.split()
        if bagian:
            out.add(int(float(bagian[0])))
    return out


def baca_nama_kelas(split_dir):
    import yaml
    d = yaml.safe_load((Path(split_dir) / "data.yaml").read_text(encoding="utf-8"))
    names = d["names"]
    return names if isinstance(names, dict) else {i: n for i, n in enumerate(names)}


def tulis_yaml_test(folder, nama):
    isi = [f"path: {Path(folder).resolve().as_posix()}", "train: images/test", "val: images/test",
           "test: images/test", "names:"]
    isi += [f"  {i}: {n}" for i, n in sorted(nama.items())]
    yaml_path = Path(folder) / "data.yaml"
    yaml_path.write_text("\n".join(isi) + "\n", encoding="utf-8")
    return yaml_path


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def metrik(r):
    d = {"mask_p": float(r.seg.mp), "mask_r": float(r.seg.mr), "mask_map50": float(r.seg.map50),
         "mask_map5095": float(r.seg.map), "box_p": float(r.box.mp), "box_r": float(r.box.mr),
         "box_map50": float(r.box.map50), "box_map5095": float(r.box.map)}
    d["mask_f1"] = f1(d["mask_p"], d["mask_r"])
    d["box_f1"] = f1(d["box_p"], d["box_r"])
    return d


KOLOM_METRIK = ["mask_p", "mask_r", "mask_f1", "mask_map50", "mask_map5095",
                "box_p", "box_r", "box_f1", "box_map50", "box_map5095"]


# --------------------------------------------------------------------------- 1. evaluasi total + per kelas
def evaluasi_total(model, yaml_path, args, out):
    r = model.val(data=str(yaml_path), split="test", imgsz=args.imgsz, batch=args.batch, device=args.device,
                  plots=True, verbose=False, project=str(out.resolve()), name="evaluasi_val", exist_ok=True)
    total = metrik(r)
    total["speed_ms_ultralytics"] = {k: float(v) for k, v in r.speed.items()}
    nama = model.names
    baris = []
    nt_inst = getattr(r, "nt_per_class", None)
    nt_img = getattr(r, "nt_per_image", None)
    for i, c in enumerate(list(getattr(r, "ap_class_index", []))):
        rec = {"kelas": nama[c]}
        for jenis, m in (("mask", r.seg), ("box", r.box)):
            try:
                p, rc, ap50, ap = m.class_result(i)
            except Exception:  # versi ultralytics berbeda
                p = rc = ap50 = float("nan")
                ap = float(m.maps[c])
            rec.update({f"{jenis}_p": float(p), f"{jenis}_r": float(rc), f"{jenis}_f1": f1(float(p), float(rc)),
                        f"{jenis}_map50": float(ap50), f"{jenis}_map5095": float(ap)})
        rec["n_gambar"] = int(nt_img[c]) if nt_img is not None else ""
        rec["n_instance"] = int(nt_inst[c]) if nt_inst is not None else ""
        baris.append(rec)
    return total, baris


# --------------------------------------------------------------------------- 2. per varian
def evaluasi_per_varian(model, split_dir, gambar, nama, args, out):
    grup = defaultdict(list)
    for p in gambar:
        grup[nama_varian(p.stem)].append(p)
    asli = grup.get("asli", [])
    non_asli = [p for k, v in grup.items() if k != "asli" for p in v]
    kumpulan = dict(sorted(grup.items()))
    if asli and non_asli:
        kumpulan = {"ASLI": asli, "SEMUA_VARIAN": non_asli, **kumpulan}

    tmp = out / "_tmp_varian"
    hasil = []
    for nama_g, daftar in kumpulan.items():
        if len(daftar) < 3:
            hasil.append({"varian": nama_g, "n_gambar": len(daftar), "catatan": "terlalu sedikit (<3), dilewati"})
            continue
        d = tmp / nama_g
        if d.exists():
            shutil.rmtree(d)
        (d / "images" / "test").mkdir(parents=True)
        (d / "labels" / "test").mkdir(parents=True)
        for p in daftar:
            shutil.copy2(p, d / "images" / "test" / p.name)
            lbl = Path(split_dir) / "labels" / "test" / f"{p.stem}.txt"
            (d / "labels" / "test" / lbl.name).write_text(lbl.read_text() if lbl.exists() else "")
        yaml_path = tulis_yaml_test(d, nama)
        try:
            r = model.val(data=str(yaml_path), split="test", imgsz=args.imgsz, batch=args.batch,
                          device=args.device, plots=False, verbose=False,
                          project=str((out / "_tmp_val").resolve()), name=nama_g, exist_ok=True)
            hasil.append({"varian": nama_g, "n_gambar": len(daftar), **metrik(r), "catatan": ""})
        except Exception as e:  # satu kelompok gagal tidak menghentikan yang lain
            hasil.append({"varian": nama_g, "n_gambar": len(daftar), "catatan": f"gagal: {e}"})
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(out / "_tmp_val", ignore_errors=True)
    return hasil


# --------------------------------------------------------------------------- 3. kesalahan tingkat gambar + contoh
def analisis_kesalahan(model, split_dir, gambar, args, out):
    salah, benar = [], []
    nama = model.names
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for p in gambar:
        gt = kelas_gt(split_dir, p.stem)
        res = model.predict(str(p), imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
        pred = set(int(c) for c in res.boxes.cls.tolist()) if res.boxes is not None and len(res.boxes) else set()
        terlewat, nyasar = gt - pred, pred - gt
        for c in gt & pred:
            tp[c] += 1
        for c in nyasar:
            fp[c] += 1
        for c in terlewat:
            fn[c] += 1
        rec = {"file_gambar": p.name, "varian": nama_varian(p.stem),
               "kelas_asli": ";".join(nama[c] for c in sorted(gt)),
               "kelas_prediksi": ";".join(nama[c] for c in sorted(pred)),
               "terlewat": ";".join(nama[c] for c in sorted(terlewat)),
               "nyasar": ";".join(nama[c] for c in sorted(nyasar))}
        (salah if (terlewat or nyasar) else benar).append((rec, p, res))

    with open(out / "kesalahan_gambar.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file_gambar", "varian", "kelas_asli", "kelas_prediksi", "terlewat", "nyasar"])
        w.writeheader()
        w.writerows(r for r, _, _ in salah)

    import cv2
    rng = random.Random(0)
    for sub, daftar in (("salah", salah), ("benar", benar)):
        d = out / "contoh_prediksi" / sub
        d.mkdir(parents=True, exist_ok=True)
        pilih = daftar if len(daftar) <= args.contoh else rng.sample(daftar, args.contoh)
        for rec, p, res in pilih:
            cv2.imwrite(str(d / p.name), res.plot())
    per_varian_salah = defaultdict(int)
    for rec, _, _ in salah:
        per_varian_salah[rec["varian"]] += 1
    kelas_semua = sorted(set(tp) | set(fp) | set(fn))
    tabel = []
    for c in kelas_semua:
        pr = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        rc = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        tabel.append({"kelas": nama[c], "TP": tp[c], "FP": fp[c], "FN": fn[c],
                      "presisi_gambar": pr, "recall_gambar": rc, "f1_gambar": f1(pr, rc)})
    tulis_csv(out / "tingkat_gambar_per_kelas.csv", tabel,
              ["kelas", "TP", "FP", "FN", "presisi_gambar", "recall_gambar", "f1_gambar"])
    TP, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    mp = TP / (TP + FP) if TP + FP else 0.0
    mr = TP / (TP + FN) if TP + FN else 0.0
    return {"gambar_test": len(gambar), "salah_tingkat_gambar": len(salah),
            "akurasi_tingkat_gambar": round(1 - len(salah) / len(gambar), 4),
            "presisi_mikro_gambar": round(mp, 4), "recall_mikro_gambar": round(mr, 4),
            "f1_mikro_gambar": round(f1(mp, mr), 4),
            "f1_makro_gambar": round(statistics.fmean([t["f1_gambar"] for t in tabel]), 4) if tabel else 0.0,
            "salah_per_varian": dict(per_varian_salah), "per_kelas": tabel}


# --------------------------------------------------------------------------- 4. kecepatan
def ukur_kecepatan(model, gambar, args):
    n = min(args.n_kecepatan, len(gambar))
    sampel = gambar[:n]
    for p in sampel[:5]:  # pemanasan
        model.predict(str(p), imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
    waktu = []
    for p in sampel:
        t0 = time.perf_counter()
        model.predict(str(p), imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
        waktu.append((time.perf_counter() - t0) * 1000)
    rata = statistics.fmean(waktu)
    return {"n_gambar": n, "ms_rata2": round(rata, 2), "ms_median": round(statistics.median(waktu), 2),
            "ms_maks": round(max(waktu), 2), "fps_rata2": round(1000 / rata, 1),
            "catatan": "diukur di perangkat ini (batch 1, termasuk pra/pasca-proses), bukan di Raspberry Pi"}


def tulis_csv(path, baris, kolom):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=kolom, extrasaction="ignore")
        w.writeheader()
        for b in baris:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in b.items()})


def _batang(ax, label, seri, judul, ylim=(0, 1.2)):
    """Diagram batang berkelompok; nilai ditulis di atas tiap batang."""
    n = len(seri)
    lebar = 0.8 / n
    for i, (nm, vals) in enumerate(seri.items()):
        xs = [j - 0.4 + lebar * (i + 0.5) for j in range(len(label))]
        bars = ax.bar(xs, [v if v == v else 0 for v in vals], lebar, label=nm)
        if len(label) * n <= 60:
            for b_, v in zip(bars, vals):
                if v == v:
                    ax.text(b_.get_x() + b_.get_width() / 2, b_.get_height() + 0.01, f"{v:.2f}",
                            ha="center", va="bottom", fontsize=7, rotation=90 if len(label) * n > 20 else 0)
    ax.set_xticks(range(len(label)))
    ax.set_xticklabels(label, rotation=30 if len(label) > 4 else 0, ha="right" if len(label) > 4 else "center")
    ax.set_ylim(*ylim)
    ax.set_title(judul)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=min(n, 5), loc="upper center")


def buat_grafik(out, total, per_kelas, per_varian=None, kls_gambar=None, judul=""):
    """Simpan grafik indikator akurasi (PNG) ke <out>/grafik/. total: kunci tanpa awalan 'test_' (mask_p, box_map5095, ...)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib tidak ada: grafik dilewati (pip install matplotlib).")
        return []
    d = Path(out) / "grafik"
    d.mkdir(parents=True, exist_ok=True)
    ind = [("p", "Presisi"), ("r", "Recall"), ("f1", "F1"), ("map50", "mAP50"), ("map5095", "mAP50-95")]
    nama_ind = [n for _, n in ind]
    hasil = []

    def simpan(fig, nama):
        fig.tight_layout()
        fig.savefig(d / nama, dpi=150)
        plt.close(fig)
        hasil.append(d / nama)

    fig, ax = plt.subplots(figsize=(8, 5))
    _batang(ax, nama_ind, {"Mask": [total.get(f"mask_{k}", float("nan")) for k, _ in ind],
                           "Kotak": [total.get(f"box_{k}", float("nan")) for k, _ in ind]},
            f"Indikator akurasi total (test asli) {judul}")
    simpan(fig, "1_ringkasan_total.png")

    if per_kelas:
        kelas = [b["kelas"] for b in per_kelas]
        fig, axs = plt.subplots(2, 1, figsize=(max(8, 1.3 * len(kelas) + 3), 10))
        for ax, j, lab in ((axs[0], "mask", "Mask"), (axs[1], "box", "Kotak")):
            _batang(ax, kelas, {n: [b.get(f"{j}_{k}", float("nan")) for b in per_kelas] for k, n in ind},
                    f"Akurasi per kelas - {lab} (test asli)")
        simpan(fig, "2_per_kelas.png")

    if per_varian:
        pv = [b for b in per_varian if "mask_map5095" in b]
        if pv:
            fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(pv) + 3), 5.5))
            _batang(ax, [f"{b['varian']} (n={b['n_gambar']})" for b in pv],
                    {n: [b.get(f"mask_{k}", float("nan")) for b in pv] for k, n in ind},
                    "Akurasi mask per varian gambar (test asli)")
            simpan(fig, "3_per_varian.png")

    if kls_gambar:
        fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(kls_gambar) + 3), 5))
        _batang(ax, [t["kelas"] for t in kls_gambar],
                {"Presisi": [t["presisi_gambar"] for t in kls_gambar], "Recall": [t["recall_gambar"] for t in kls_gambar],
                 "F1": [t["f1_gambar"] for t in kls_gambar]}, "Akurasi tingkat-gambar per kelas")
        simpan(fig, "4_tingkat_gambar.png")
    print("Grafik: " + ", ".join(p.name for p in hasil) + f"  -> {d}")
    return hasil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", type=Path, default=ANN_DIR, help="folder berisi split_r*_s*/")
    ap.add_argument("--hasil07", type=Path, default=(HASIL / "07_latih_final") if HASIL else Path("hasil_07"),
                    help="folder output tahap 7 (berisi pilihan_final.json)")
    ap.add_argument("--split", default=None, help="nama split, mis. split_r80_s44 (default: dari pilihan_final.json)")
    ap.add_argument("--model", type=Path, default=Path(__file__).resolve().parent / "models" / "best.pt")
    ap.add_argument("--out", type=Path, default=(HASIL / "08_evaluasi_final") if HASIL else Path("hasil_08"))
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--conf", type=float, default=0.25, help="ambang keyakinan untuk prediksi gambar/kesalahan")
    ap.add_argument("--contoh", type=int, default=20, help="maks gambar contoh per folder (salah/benar)")
    ap.add_argument("--n-kecepatan", type=int, default=100, help="jumlah gambar untuk ukur kecepatan")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.splits_dir is None:
        raise SystemExit("Tentukan --splits-dir (folder berisi split_r*_s*/).")
    if not args.model.exists():
        raise SystemExit(f"{args.model} tidak ditemukan. Jalankan 07_latih_final.py dulu.")
    nama_split = args.split
    if not nama_split:
        pf = args.hasil07 / "pilihan_final.json"
        if not pf.exists():
            raise SystemExit(f"{pf} tidak ada. Beri --split split_rXX_sYY secara manual.")
        nama_split = json.loads(pf.read_text(encoding="utf-8"))["split"]
    split_dir = Path(args.splits_dir) / nama_split
    if not (split_dir / "data.yaml").exists():
        raise SystemExit(f"Split {split_dir} tidak ditemukan.")
    args.out.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(str(args.model))
    nama = baca_nama_kelas(split_dir)
    gambar = daftar_test(split_dir)
    print(f"Model: {args.model}\nSplit: {nama_split} | gambar test: {len(gambar)}")

    # yaml sementara yang menunjuk test asli
    yaml_test = args.out / "data_test.yaml"
    isi = [f"path: {split_dir.resolve().as_posix()}", "train: images/test", "val: images/test", "test: images/test",
           "names:"] + [f"  {i}: {n}" for i, n in sorted(nama.items())]
    yaml_test.write_text("\n".join(isi) + "\n", encoding="utf-8")

    print("\n[1/4] Evaluasi total + per kelas ...")
    total, per_kelas = evaluasi_total(model, yaml_test, args, args.out)
    tulis_csv(args.out / "per_kelas.csv", per_kelas, ["kelas", "n_gambar", "n_instance"] + KOLOM_METRIK)

    print("[2/4] Evaluasi per varian (gerak/cahaya/dll.) ...")
    per_varian = evaluasi_per_varian(model, split_dir, gambar, nama, args, args.out)
    tulis_csv(args.out / "per_varian.csv", per_varian,
              ["varian", "n_gambar"] + KOLOM_METRIK + ["catatan"])

    print("[3/4] Analisis kesalahan + contoh prediksi ...")
    kesalahan = analisis_kesalahan(model, split_dir, gambar, args, args.out)

    print("[4/4] Ukur kecepatan ...")
    kecepatan = ukur_kecepatan(model, gambar, args)
    (args.out / "kecepatan.json").write_text(json.dumps(kecepatan, indent=1, ensure_ascii=False), encoding="utf-8")

    (args.out / "ringkasan_test.json").write_text(
        json.dumps({"model": str(args.model), "split": nama_split, "total": total, "kesalahan": kesalahan,
                    "kecepatan": kecepatan}, indent=1, ensure_ascii=False), encoding="utf-8")

    L = []
    L.append("RINGKASAN AKURASI (test asli)")
    L.append(f"Model: {args.model} | Split: {nama_split} | gambar test: {len(gambar)}")
    L.append("")
    L.append(f"{'':<6}{'P':>8}{'R':>8}{'F1':>8}{'mAP50':>8}{'mAP50-95':>10}")
    for jenis, lab in (("mask", "MASK"), ("box", "KOTAK")):
        L.append(f"{lab:<6}{total[jenis+'_p']:>8.4f}{total[jenis+'_r']:>8.4f}{total[jenis+'_f1']:>8.4f}"
                 f"{total[jenis+'_map50']:>8.4f}{total[jenis+'_map5095']:>10.4f}")
    L.append("")
    L.append(f"Tingkat-gambar (conf>={args.conf}): akurasi {kesalahan['akurasi_tingkat_gambar']:.4f} "
             f"({kesalahan['gambar_test'] - kesalahan['salah_tingkat_gambar']}/{kesalahan['gambar_test']} benar) | "
             f"P {kesalahan['presisi_mikro_gambar']:.4f} R {kesalahan['recall_mikro_gambar']:.4f} "
             f"F1-mikro {kesalahan['f1_mikro_gambar']:.4f} F1-makro {kesalahan['f1_makro_gambar']:.4f}")
    L.append(f"Kecepatan di perangkat ini: {kecepatan['ms_rata2']} ms/gambar (~{kecepatan['fps_rata2']} FPS)")
    L.append("")
    L.append("--- PER KELAS (mask | kotak) ---")
    L.append(f"{'kelas':<14}{'nImg':>5}{'nInst':>6} |{'P':>7}{'R':>7}{'F1':>7}{'mAP50':>7}{'50-95':>7} |"
             f"{'P':>7}{'R':>7}{'F1':>7}{'mAP50':>7}{'50-95':>7}")
    for b_ in per_kelas:
        L.append(f"{b_['kelas']:<14}{str(b_['n_gambar']):>5}{str(b_['n_instance']):>6} |"
                 + "".join(f"{b_['mask_'+k]:>7.3f}" for k in ("p", "r", "f1", "map50", "map5095")) + " |"
                 + "".join(f"{b_['box_'+k]:>7.3f}" for k in ("p", "r", "f1", "map50", "map5095")))
    L.append("")
    L.append("--- PER VARIAN (mask | kotak) ---")
    L.append(f"{'varian':<20}{'n':>5} |{'P':>7}{'R':>7}{'F1':>7}{'mAP50':>7}{'50-95':>7} |"
             f"{'P':>7}{'R':>7}{'F1':>7}{'mAP50':>7}{'50-95':>7}")
    for b_ in per_varian:
        if "mask_map5095" in b_:
            L.append(f"{b_['varian']:<20}{b_['n_gambar']:>5} |"
                     + "".join(f"{b_['mask_'+k]:>7.3f}" for k in ("p", "r", "f1", "map50", "map5095")) + " |"
                     + "".join(f"{b_['box_'+k]:>7.3f}" for k in ("p", "r", "f1", "map50", "map5095")))
        else:
            L.append(f"{b_['varian']:<20}{b_['n_gambar']:>5}   ({b_.get('catatan', '')})")
    L.append("")
    L.append("--- PER KELAS TINGKAT-GAMBAR ---")
    L.append(f"{'kelas':<14}{'TP':>5}{'FP':>5}{'FN':>5}{'P':>8}{'R':>8}{'F1':>8}")
    for t in kesalahan["per_kelas"]:
        L.append(f"{t['kelas']:<14}{t['TP']:>5}{t['FP']:>5}{t['FN']:>5}{t['presisi_gambar']:>8.3f}"
                 f"{t['recall_gambar']:>8.3f}{t['f1_gambar']:>8.3f}")
    laporan = "\n".join(L)
    print("\n" + laporan)
    (args.out / "laporan_akurasi.txt").write_text(laporan + "\n", encoding="utf-8")
    buat_grafik(args.out, total, per_kelas, per_varian, kesalahan["per_kelas"], judul=f"- {nama_split}")
    print(f"\nSemua hasil: {args.out}")
    print("Lanjut: export ke NCNN untuk Raspberry Pi (tahap 9).")


if __name__ == "__main__":
    main()