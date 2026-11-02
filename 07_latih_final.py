"""TAHAP 7 - LATIH FINAL: ambil pemenang dari tahap 6, latih ulang pakai train+val, simpan models/best.pt.

Alur:
  1. Baca hasil tahap 6 (ringkasan_rasio.csv) -> pilih RASIO terbaik (rata-rata skor test lintas model & seed)
     dan MODEL terbaik pada rasio itu. Bisa dioverride manual (--model / --rasio / --seed).
  2. Pilih SEED split yang representatif: seed yang skor test-nya paling dekat MEDIAN
     (bukan yang tertinggi, supaya tidak memilih split yang kebetulan paling beruntung).
  3. Gabungkan train + val dari split itu, lalu sisihkan val-internal kecil (default 10%)
     per KELOMPOK (varian gerak/cahaya dari gambar sumber yang sama + frame berurutan tidak dipisah),
     supaya tidak ada kebocoran antara train final dan val-internal.
  4. Latih ulang model pemenang dari bobot COCO. TEST ASLI TIDAK PERNAH dipakai latih/val.
  5. Evaluasi best.pt di test asli, simpan laporan, salin bobot ke models/best.pt
     (best.pt lama DIARSIPKAN, tidak dihapus).

Input  : hasil tahap 5 (folder split_r*_s*/) dan tahap 6 (06_perbandingan_model/)
Output : data/hasil/07_latih_final/
           dataset/            data gabungan (train final, val-internal) + data.yaml (test menunjuk ke test asli)
           runs/               log training ultralytics
           pilihan_final.json  alasan pemilihan model/rasio/seed
           laporan_final.json  skor test model final (+ per kelas, mask & kotak)
           laporan_akurasi.txt, per_kelas_test.csv  ringkasan lengkap P/R/F1/mAP
           evaluasi_test/      confusion matrix & kurva PR di test asli
         models/best.pt        bobot final untuk export ke Raspberry Pi (tahap 9)

Jalankan (lokal):
  python scripts/07_latih_final.py
  python scripts/07_latih_final.py --epochs 3 --patience 3          # uji cepat
  python scripts/07_latih_final.py --model yolo26n-seg --rasio 80   # tentukan manual

Kalau memakai cara bertahap (cari rasio dulu dengan satu model, baru bandingkan model):
  python scripts/07_latih_final.py --file-rasio hasil/ringkasan_rasio_tahapA.csv \\
                                   --file-model hasil/ringkasan_rasio.csv

Jalankan (Colab, setelah unzip dataset_colab.zip ke /content dan menjalankan 06):
  !python 07_latih_final.py --splits-dir /content/dataset --hasil06 /content/hasil_06 \\
          --out /content/hasil_07 --models-dir /content/models
"""
import argparse
import csv
import json
import random
import re
import shutil
import statistics
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    from config import ANN_DIR, HASIL
except ImportError:  # dijalankan di Colab tanpa config
    ANN_DIR = HASIL = None

EKSTENSI = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
METRIK = "test_mask_map5095"


# --------------------------------------------------------------------------- util umum
def arsipkan(path):
    """Pindahkan file/folder lama ke _arsip/ di sebelahnya (tidak dihapus). Return path arsip atau None."""
    path = Path(path)
    if not path.exists():
        return None
    arsip = path.parent / "_arsip"
    arsip.mkdir(exist_ok=True)
    tujuan = arsip / f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}"
    shutil.move(str(path), str(tujuan))
    return tujuan


def baca_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def angka(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# --------------------------------------------------------------------------- 1. pilih pemenang
def pilih_pemenang(file_rasio, file_model, model_paksa=None, rasio_paksa=None):
    """Return (model_stem, rasio, catatan:list[str], tabel:dict)."""
    catatan = []
    rows_r = baca_csv(file_rasio)
    rows_m = baca_csv(file_model)
    kunci = f"{METRIK}_mean"
    kunci_sd = f"{METRIK}_std"
    if not rows_r or kunci not in rows_r[0]:
        raise SystemExit(f"{file_rasio} kosong / tidak punya kolom {kunci}. Jalankan tahap 6 dulu.")

    # ---- rasio: rata-rata skor lintas model per rasio ----
    if rasio_paksa is not None:
        rasio = int(rasio_paksa)
        catatan.append(f"rasio ditentukan manual: {rasio}")
    else:
        per_rasio = defaultdict(list)
        for r in rows_r:
            if model_paksa and r["model"] != model_paksa:
                continue
            per_rasio[int(r["rasio"])].append((angka(r[kunci]), angka(r[kunci_sd])))
        if not per_rasio:
            raise SystemExit(f"Tidak ada baris untuk model '{model_paksa}' di {file_rasio}.")
        skor = {k: (statistics.fmean(a for a, _ in v), statistics.fmean(s for _, s in v))
                for k, v in per_rasio.items()}
        urut = sorted(skor.items(), key=lambda x: x[1][0], reverse=True)
        rasio = urut[0][0]
        catatan.append("ranking rasio (rata-rata test mask mAP50-95): " +
                       ", ".join(f"r{k}={m:.4f}" for k, (m, _) in urut))
        if len(urut) > 1:
            (r1, (m1, s1)), (r2, (m2, s2)) = urut[0], urut[1]
            if m1 - m2 < max(s1, s2):
                catatan.append(f"PERINGATAN: selisih rasio {r1} vs {r2} ({m1 - m2:.4f}) lebih kecil dari simpangan "
                               f"baku antar seed ({max(s1, s2):.4f}). Rasio ini bukan pemenang yang tegas; "
                               "pertimbangkan rasio dengan test set lebih besar.")

    # ---- model: terbaik pada rasio terpilih ----
    if model_paksa:
        model = model_paksa
        catatan.append(f"model ditentukan manual: {model}")
    else:
        kandidat = [r for r in rows_m if int(r["rasio"]) == rasio] or rows_m
        if len(kandidat) != len([r for r in rows_m if int(r["rasio"]) == rasio]):
            catatan.append(f"PERINGATAN: tidak ada hasil rasio {rasio} di {file_model}; model dipilih dari semua rasio.")
        urut = sorted(kandidat, key=lambda r: angka(r[kunci]), reverse=True)
        model = urut[0]["model"]
        catatan.append("ranking model pada rasio terpilih: " +
                       ", ".join(f"{r['model']}={angka(r[kunci]):.4f}" for r in urut))
        if len(urut) > 1:
            gap = angka(urut[0][kunci]) - angka(urut[1][kunci])
            sd = max(angka(urut[0][kunci_sd]), angka(urut[1][kunci_sd]))
            if gap < sd:
                catatan.append(f"PERINGATAN: selisih {urut[0]['model']} vs {urut[1]['model']} ({gap:.4f}) lebih kecil "
                               f"dari simpangan baku antar seed ({sd:.4f}). Jangan diklaim unggul di laporan.")
    return model, rasio, catatan


def pilih_seed_representatif(dir_json, model, rasio, seed_paksa=None):
    """Seed yang skor test-nya paling dekat median untuk (model, rasio)."""
    if seed_paksa is not None:
        return int(seed_paksa), "seed ditentukan manual"
    skor = {}
    for p in sorted(Path(dir_json).glob("*.json")):
        d = json.loads(p.read_text())
        if d.get("model") == model and int(d.get("rasio", -1)) == rasio and METRIK in d:
            skor[int(d["seed_split"])] = float(d[METRIK])
    if not skor:
        raise SystemExit(f"Tidak ada hasil run untuk {model} rasio {rasio} di {dir_json}. "
                         "Gunakan --seed untuk menentukan seed split manual.")
    med = statistics.median(skor.values())
    seed = min(skor, key=lambda s: (abs(skor[s] - med), s))
    ket = (f"seed paling dekat median: s{seed} (skor {skor[seed]:.4f}, median {med:.4f}; "
           + ", ".join(f"s{s}={v:.4f}" for s, v in sorted(skor.items())) + ")")
    return seed, ket


# --------------------------------------------------------------------------- 2. bangun data final
def kunci_kelompok(stem):
    """Kelompok = gambar sumber (buang akhiran __varian) tanpa nomor frame di ujung.
    Semua varian + frame berurutan dari video yang sama jatuh di kelompok yang sama."""
    sumber = stem.split("__")[0]
    return re.sub(r"[_\-]?(?:frame|f|img|image)?[_\-]?\d+$", "", sumber, flags=re.I) or sumber


def daftar_gambar(split_dir, bagian):
    d = Path(split_dir) / "images" / bagian
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in EKSTENSI)


def pilih_dekat(kelompok, target, rng):
    """Tambah kelompok hanya kalau total makin dekat ke target (sama gagasan dengan tahap 5)."""
    kunci = sorted(kelompok)
    rng.shuffle(kunci)
    dipilih, total = [], 0
    for k in kunci:
        n = len(kelompok[k])
        if abs(total + n - target) < abs(total - target):
            dipilih.append(k)
            total += n
    if not dipilih and kunci:  # target sangat kecil: ambil kelompok terkecil
        dipilih = [min(kunci, key=lambda k: len(kelompok[k]))]
    return set(dipilih)


def pilih_val_stratifikasi(kelompok, asal, sumber_lbl, target, seed):
    """Pilih kelompok val-internal sambil memastikan setiap kelas terwakili."""
    kelas_per_kelompok = defaultdict(set)
    for k, gambar in kelompok.items():
        for gambar_path in gambar:
            label_path = sumber_lbl[asal[gambar_path]] / f"{gambar_path.stem}.txt"
            if not label_path.exists():
                continue
            for baris in label_path.read_text(encoding="utf-8").splitlines():
                bagian = baris.split()
                if bagian:
                    kelas_per_kelompok[k].add(int(float(bagian[0])))

    wajib = set()
    for kelas in sorted({c for nilai in kelas_per_kelompok.values() for c in nilai}):
        kandidat = [k for k, nilai in kelas_per_kelompok.items() if kelas in nilai]
        if kandidat:
            wajib.add(min(kandidat, key=lambda k: (len(kelompok[k]), k)))

    terpilih = set(wajib)
    sisa_target = max(0, target - sum(len(kelompok[k]) for k in terpilih))
    sisanya = {k: v for k, v in kelompok.items() if k not in terpilih}
    terpilih.update(pilih_dekat(sisanya, sisa_target, random.Random(seed)))
    return terpilih


def salin(gambar, label_dir_sumber, img_out, lbl_out):
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(gambar, img_out / gambar.name)
    lbl = Path(label_dir_sumber) / f"{gambar.stem}.txt"
    if lbl.exists():
        shutil.copy2(lbl, lbl_out / lbl.name)
    else:
        (lbl_out / lbl.name).write_text("")  # gambar tanpa objek


def baca_nama_kelas(split_dir):
    import yaml
    d = yaml.safe_load((Path(split_dir) / "data.yaml").read_text(encoding="utf-8"))
    names = d["names"]
    return names if isinstance(names, dict) else {i: n for i, n in enumerate(names)}


def evaluasi_tingkat_gambar(model, gambar, label_dir, nama, args, out):
    """Hitung TP/FP/FN/TN per kelas berdasarkan ada/tidaknya kelas di tiap gambar test."""
    hitung = {c: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for c in nama}
    for gambar_path in gambar:
        aktual = set()
        label_path = Path(label_dir) / f"{gambar_path.stem}.txt"
        if label_path.exists():
            for baris in label_path.read_text(encoding="utf-8").splitlines():
                bagian = baris.split()
                if bagian:
                    aktual.add(int(float(bagian[0])))
        hasil = model.predict(str(gambar_path), imgsz=args.imgsz, conf=0.25,
                              batch=1, device=args.device, verbose=False)[0]
        prediksi = set()
        if hasil.boxes is not None:
            prediksi = {int(c) for c in hasil.boxes.cls.tolist()}
        for c in nama:
            a, p = c in aktual, c in prediksi
            k = "TP" if a and p else "FP" if p else "FN" if a else "TN"
            hitung[c][k] += 1

    tabel = []
    for c, nilai in hitung.items():
        tp, fp, fn, tn = (nilai[k] for k in ("TP", "FP", "FN", "TN"))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        tabel.append({"kelas": nama[c], **nilai,
                      "akurasi_gambar": (tp + tn) / len(gambar) if gambar else 0.0,
                      "presisi_gambar": p, "recall_gambar": r, "f1_gambar": f1(p, r)})

    path_csv = Path(out) / "konfusi_tingkat_gambar.csv"
    with path_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["kelas", "TP", "FP", "FN", "TN", "akurasi_gambar",
                                          "presisi_gambar", "recall_gambar", "f1_gambar"])
        w.writeheader()
        w.writerows(tabel)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, len(tabel), figsize=(4 * max(1, len(tabel)), 4))
        axs = [axs] if len(tabel) == 1 else axs
        for ax, rec in zip(axs, tabel):
            mat = [[rec["TN"], rec["FP"]], [rec["FN"], rec["TP"]]]
            ax.imshow(mat, cmap="Blues")
            for y in range(2):
                for x in range(2):
                    ax.text(x, y, mat[y][x], ha="center", va="center")
            ax.set_title(rec["kelas"])
            ax.set_xticks([0, 1], ["Negatif", "Positif"])
            ax.set_yticks([0, 1], ["Negatif", "Positif"])
            ax.set_xlabel("Prediksi")
            ax.set_ylabel("Aktual")
        fig.tight_layout()
        fig.savefig(Path(out) / "konfusi_tingkat_gambar.png", dpi=150)
        plt.close(fig)
    except ImportError:
        print("matplotlib tidak ada: matriks tingkat gambar dilewati.")

    return tabel


def bangun_data_final(split_dir, out_dataset, val_frac, seed):
    split_dir = Path(split_dir)
    gambar = daftar_gambar(split_dir, "train") + daftar_gambar(split_dir, "val")
    if not gambar:
        raise SystemExit(f"Tidak ada gambar train/val di {split_dir}.")
    sumber_lbl = {"train": split_dir / "labels" / "train", "val": split_dir / "labels" / "val"}
    asal = {p: ("train" if p.parent.name == "train" else "val") for p in gambar}

    kelompok = defaultdict(list)
    for p in gambar:
        kelompok[kunci_kelompok(p.stem)].append(p)
    target = max(1, round(len(gambar) * val_frac))
    val_kel = pilih_val_stratifikasi(kelompok, asal, sumber_lbl, target, seed)

    if out_dataset.exists():
        t = arsipkan(out_dataset)
        print(f"  dataset lama diarsipkan: {t}")
    n = {"train": 0, "val": 0}
    for k, daftar in kelompok.items():
        bagian = "val" if k in val_kel else "train"
        for p in daftar:
            salin(p, sumber_lbl[asal[p]], out_dataset / "images" / bagian, out_dataset / "labels" / bagian)
            n[bagian] += 1

    # kebocoran antar kelompok mustahil (kelompok utuh pindah sisi), tapi dicek sebagai pengaman
    kt = {kunci_kelompok(p.stem) for p in (out_dataset / "images" / "train").iterdir()}
    kv = {kunci_kelompok(p.stem) for p in (out_dataset / "images" / "val").iterdir()}
    assert not (kt & kv), "kelompok bocor antara train dan val-internal"

    test_asli = (split_dir / "images" / "test").resolve()
    nama = baca_nama_kelas(split_dir)
    yaml_path = out_dataset / "data.yaml"
    isi = [f"path: {out_dataset.resolve().as_posix()}", "train: images/train", "val: images/val"]
    if test_asli.is_dir():
        isi.append(f"test: {test_asli.as_posix()}")
    isi.append("names:")
    isi += [f"  {i}: {n_}" for i, n_ in sorted(nama.items())]
    yaml_path.write_text("\n".join(isi) + "\n", encoding="utf-8")
    return yaml_path, n, len(kelompok), len(val_kel), nama


# --------------------------------------------------------------------------- 3. latih + evaluasi
def latih_dan_evaluasi(bobot, yaml_path, test_dir, nama_kelas, args, out, nama_run):
    from ultralytics import YOLO
    checkpoint = Path(out) / "runs" / nama_run / "weights" / "last.pt"
    if args.resume:
        if not checkpoint.exists():
            raise SystemExit(f"Checkpoint untuk resume tidak ditemukan: {checkpoint}")
        bobot_latih = str(checkpoint)
    else:
        bobot_latih = bobot
    t0 = time.time()
    model = YOLO(bobot_latih)
    model.train(data=str(yaml_path), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                patience=args.patience, seed=args.seed_latih, deterministic=True, device=args.device,
                workers=args.workers, project=str((out / "runs").resolve()), name=nama_run, exist_ok=True,
                resume=args.resume, verbose=False)
    waktu = (time.time() - t0) / 60
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    m = YOLO(str(best))
    laporan = {"bobot_awal": bobot, "epoch": args.epochs, "waktu_latih_menit": round(waktu, 2),
               "ukuran_mb": round(best.stat().st_size / 1e6, 2),
               "parameter_juta": round(sum(p.numel() for p in m.model.parameters()) / 1e6, 3)}
    r = m.val(data=str(yaml_path), split="test", imgsz=args.imgsz, batch=args.batch, device=args.device,
              plots=True, verbose=False, project=str((out / "evaluasi_test").resolve()), name=nama_run, exist_ok=True)
    laporan.update({"ms_preprocess": float(r.speed.get("preprocess", float("nan"))),
                    "ms_inferensi": float(r.speed.get("inference", float("nan"))),
                    "ms_postprocess": float(r.speed.get("postprocess", float("nan")))})
    for jenis, mt in (("box", r.box), ("mask", r.seg)):
        p, rc = float(mt.mp), float(mt.mr)
        laporan.update({f"test_{jenis}_p": p, f"test_{jenis}_r": rc, f"test_{jenis}_f1": f1(p, rc),
                        f"test_{jenis}_map50": float(mt.map50), f"test_{jenis}_map5095": float(mt.map)})
    ada = list(getattr(r, "ap_class_index", []))
    nt_inst, nt_img = getattr(r, "nt_per_class", None), getattr(r, "nt_per_image", None)
    laporan["per_kelas_test_mask_map5095"] = {m.names[c]: float(r.seg.maps[c]) for c in ada}  # kompatibel dgn versi lama
    detail = {}
    for i, c in enumerate(ada):
        rec = {"n_gambar": int(nt_img[c]) if nt_img is not None else None,
               "n_instance": int(nt_inst[c]) if nt_inst is not None else None}
        for jenis, mt in (("mask", r.seg), ("box", r.box)):
            try:
                p, rc, a50, ap_ = mt.class_result(i)
            except Exception:
                p = rc = a50 = float("nan")
                ap_ = mt.maps[c]
            rec.update({f"{jenis}_p": float(p), f"{jenis}_r": float(rc), f"{jenis}_f1": f1(float(p), float(rc)),
                        f"{jenis}_map50": float(a50), f"{jenis}_map5095": float(ap_)})
        detail[m.names[c]] = rec
    laporan["per_kelas_test"] = detail
    gambar_test = daftar_gambar(test_dir.parent.parent, "test")
    tingkat_gambar = evaluasi_tingkat_gambar(
        m, gambar_test, test_dir.parent.parent / "labels" / "test", nama_kelas, args, out)
    laporan["gambar_test"] = len(gambar_test)
    laporan["per_kelas_tingkat_gambar"] = tingkat_gambar
    laporan["sumber_metrik"] = {
        "box_mask_precision_recall_map": "Ultralytics model.val() pada test asli",
        "f1": "2 * precision * recall / (precision + recall)",
        "TP_FP_FN_TN": "klasifikasi biner per kelas pada tingkat gambar dari label test dan model.predict(conf=0.25)",
        "confusion_matrix": "Ultralytics model.val(plots=True) dan matriks biner tingkat gambar",
        "kecepatan": "r.speed dari Ultralytics, dalam ms per batch pada perangkat ini",
    }
    return best, laporan


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
    ap.add_argument("--hasil06", type=Path, default=(HASIL / "06_perbandingan_model") if HASIL else Path("hasil_06"),
                    help="folder output tahap 6 (ringkasan_rasio.csv dan run_json/)")
    ap.add_argument("--file-rasio", type=Path, default=None, help="CSV untuk memilih RASIO (default: hasil06/ringkasan_rasio.csv)")
    ap.add_argument("--file-model", type=Path, default=None, help="CSV untuk memilih MODEL (default: hasil06/ringkasan_rasio.csv)")
    ap.add_argument("--out", type=Path, default=(HASIL / "07_latih_final") if HASIL else Path("hasil_07"))
    ap.add_argument("--models-dir", type=Path, default=Path(__file__).resolve().parent / "models",
                    help="tempat menyimpan best.pt final")
    ap.add_argument("--model", default=None, help="paksa model, mis. yolo26n-seg (tanpa .pt)")
    ap.add_argument("--rasio", type=int, default=None, help="paksa rasio, mis. 80")
    ap.add_argument("--seed", type=int, default=None, help="paksa seed split, mis. 44")
    ap.add_argument("--val-internal", type=float, default=0.10, help="porsi val-internal dari train+val (default 0.10)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seed-latih", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="lanjutkan dari runs/<nama_run>/weights/last.pt")
    args = ap.parse_args()

    if args.splits_dir is None:
        raise SystemExit("Tentukan --splits-dir (folder berisi split_r*_s*/).")
    file_rasio = args.file_rasio or args.hasil06 / "ringkasan_rasio.csv"
    file_model = args.file_model or args.hasil06 / "ringkasan_rasio.csv"
    for f in (file_rasio, file_model):
        if not Path(f).exists():
            raise SystemExit(f"{f} tidak ditemukan. Jalankan 06_bandingkan_model.py dulu.")
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- 1-2. pilih model, rasio, seed ----
    model, rasio, catatan = pilih_pemenang(file_rasio, file_model, args.model, args.rasio)
    seed, ket_seed = pilih_seed_representatif(args.hasil06 / "run_json", model, rasio, args.seed)
    catatan.append(ket_seed)
    split_dir = Path(args.splits_dir) / f"split_r{rasio}_s{seed}"
    if not (split_dir / "data.yaml").exists():
        raise SystemExit(f"Split {split_dir} tidak ada.")
    print("\nPILIHAN FINAL")
    print(f"model : {model}\nrasio : {rasio}\nseed  : {seed}\nsplit : {split_dir}")
    for c in catatan:
        print(f"  - {c}")
    pilihan = {"model": model, "rasio": rasio, "seed_split": seed, "split": split_dir.name, "catatan": catatan,
               "waktu": datetime.now().isoformat(timespec="seconds")}
    (args.out / "pilihan_final.json").write_text(json.dumps(pilihan, indent=1, ensure_ascii=False), encoding="utf-8")

    # ---- 3. data gabungan ----
    print("\nMenyusun data final (train+val digabung, val-internal disisihkan per kelompok)...")
    yaml_path, n, n_kel, n_kel_val, nama = bangun_data_final(split_dir, args.out / "dataset", args.val_internal, seed)
    total = n["train"] + n["val"]
    print(f"  train final: {n['train']} gambar | val-internal: {n['val']} gambar ({n['val'] / total:.1%}) | "
          f"kelompok: {n_kel} ({n_kel_val} ke val-internal)")
    print(f"  test asli  : {len(daftar_gambar(split_dir, 'test'))} gambar (tidak disentuh)")

    # ---- 4. latih + evaluasi ----
    print(f"\nMelatih {model} ...")
    best, laporan = latih_dan_evaluasi(
        f"{model}.pt", yaml_path, split_dir / "images" / "test", nama, args, args.out,
        f"final_{model}_r{rasio}_s{seed}")
    laporan.update({"model": model, "rasio": rasio, "seed_split": seed, "gambar_train": n["train"],
                    "gambar_val_internal": n["val"], "kelas": nama})
    (args.out / "laporan_final.json").write_text(json.dumps(laporan, indent=1, ensure_ascii=False), encoding="utf-8")

    # ---- 5. simpan bobot final ----
    args.models_dir.mkdir(parents=True, exist_ok=True)
    tujuan = args.models_dir / "best.pt"
    if tujuan.exists():
        print(f"best.pt lama diarsipkan: {arsipkan(tujuan)}")
    shutil.copy2(best, tujuan)

    L = ["HASIL MODEL FINAL (test asli)",
         f"{'':<6}{'P':>8}{'R':>8}{'F1':>8}{'mAP50':>8}{'mAP50-95':>10}"]
    for j, lab in (("mask", "MASK"), ("box", "KOTAK")):
        L.append(f"{lab:<6}{laporan[f'test_{j}_p']:>8.4f}{laporan[f'test_{j}_r']:>8.4f}{laporan[f'test_{j}_f1']:>8.4f}"
                 f"{laporan[f'test_{j}_map50']:>8.4f}{laporan[f'test_{j}_map5095']:>10.4f}")
    L.append(f"kecepatan: pra {laporan['ms_preprocess']:.1f} ms | inferensi {laporan['ms_inferensi']:.1f} ms | "
             f"pasca {laporan['ms_postprocess']:.1f} ms per gambar | {laporan['parameter_juta']} juta parameter | "
             f"{laporan['ukuran_mb']} MB | waktu latih {laporan['waktu_latih_menit']} menit")
    L += ["", "--- PER KELAS (mask | kotak) ---",
          f"{'kelas':<14}{'nImg':>5}{'nInst':>6} |{'P':>7}{'R':>7}{'F1':>7}{'mAP50':>7}{'50-95':>7} |"
          f"{'P':>7}{'R':>7}{'F1':>7}{'mAP50':>7}{'50-95':>7}"]
    for k, v in laporan["per_kelas_test"].items():
        L.append(f"{k:<14}{str(v['n_gambar']):>5}{str(v['n_instance']):>6} |"
                 + "".join(f"{v['mask_'+x]:>7.3f}" for x in ("p", "r", "f1", "map50", "map5095")) + " |"
                 + "".join(f"{v['box_'+x]:>7.3f}" for x in ("p", "r", "f1", "map50", "map5095")))
    L += ["", "--- POSITIF/NEGATIF TINGKAT GAMBAR ---",
          f"{'kelas':<14}{'TP':>6}{'FP':>6}{'FN':>6}{'TN':>6} |"
          f"{'Akurasi':>9}{'Presisi':>9}{'Recall':>9}{'F1':>9}"]
    for v in laporan["per_kelas_tingkat_gambar"]:
        L.append(f"{v['kelas']:<14}{v['TP']:>6}{v['FP']:>6}{v['FN']:>6}{v['TN']:>6} |"
                 f"{v['akurasi_gambar']:>9.4f}{v['presisi_gambar']:>9.4f}"
                 f"{v['recall_gambar']:>9.4f}{v['f1_gambar']:>9.4f}")
    L += ["", "Sumber metrik:"]
    L.extend(f"- {k}: {v}" for k, v in laporan["sumber_metrik"].items())
    teks = "\n".join(L)
    print("\n" + teks)
    (args.out / "laporan_akurasi.txt").write_text(teks + "\n", encoding="utf-8")
    buat_grafik(args.out, {k[5:]: v for k, v in laporan.items() if k.startswith("test_")},
                [{"kelas": k, **v} for k, v in laporan["per_kelas_test"].items()], judul=f"- {model} r{rasio} s{seed}")
    with open(args.out / "per_kelas_test.csv", "w", newline="", encoding="utf-8") as f:
        kol = ["kelas", "n_gambar", "n_instance"] + [f"{j}_{x}" for j in ("mask", "box") for x in ("p", "r", "f1", "map50", "map5095")]
        w = csv.DictWriter(f, fieldnames=kol)
        w.writeheader()
        for k, v in laporan["per_kelas_test"].items():
            w.writerow({"kelas": k, **{c: (f"{v[c]:.5f}" if isinstance(v[c], float) else v[c]) for c in kol[1:]}})
    print(f"\nBobot final : {tujuan}")
    print(f"Laporan     : {args.out / 'laporan_final.json'}")
    print("Lanjut: export ke NCNN untuk Raspberry Pi.")


if __name__ == "__main__":
    main()