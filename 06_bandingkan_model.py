"""TAHAP 6 - BANDINGKAN yolo11n-seg vs yolo26n-seg DAN UJI RASIO SPLIT (train -> val -> test).

Setiap kombinasi (model x split) dilatih dengan hiperparameter SAMA (epoch, imgsz, batch, seed
latih), lalu bobot best.pt dievaluasi pada val DAN pada test. Pemilihan model/rasio memakai skor
TEST (soal ujian yang sama untuk semua rasio pada seed split yang sama).

Output (folder --out):
  hasil_run.csv       satu baris per (model, split): mAP kotak/mask, P, R, kecepatan, parameter, waktu latih
  per_kelas.csv       mAP50-95 mask per kelas pada test
  ringkasan_rasio.csv rata-rata +- simpangan baku lintas seed untuk tiap (model, rasio)
  ringkasan_model.csv selisih yolo26n - yolo11n pada split yang sama (berpasangan)

Dapat DILANJUTKAN: run yang sudah punya JSON di <out>/run_json/ dilewati.

Jalankan (lokal):
  python scripts/06_bandingkan_model.py --epochs 100
Jalankan (Colab, setelah unzip dataset_colab.zip ke /content):
  !pip install -U ultralytics
  !python 06_bandingkan_model.py --splits-dir /content/dataset --out /content/hasil_06 --epochs 100

Catatan keadilan: kedua model memakai resep bawaan Ultralytics masing-masing (optimizer 'auto').
Nama bobot harus ada di ultralytics yang terpasang; jika yolo26n-seg.pt tidak ditemukan,
perbarui paket: pip install -U ultralytics.
"""
import argparse
import csv
import json
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

try:
    from config import ANN_DIR, HASIL
except ImportError:  # dijalankan di Colab tanpa config
    ANN_DIR = HASIL = None


def ambil(r):
    """Ekstrak metrik dari hasil model.val() segmentasi."""
    d = {"box_p": r.box.mp, "box_r": r.box.mr, "box_map50": r.box.map50, "box_map5095": r.box.map,
         "mask_p": r.seg.mp, "mask_r": r.seg.mr, "mask_map50": r.seg.map50, "mask_map5095": r.seg.map,
         "ms_inferensi": r.speed.get("inference", float("nan"))}
    d = {k: float(v) for k, v in d.items()}
    for j in ("box", "mask"):
        p, rc = d[f"{j}_p"], d[f"{j}_r"]
        d[f"{j}_f1"] = 2 * p * rc / (p + rc) if p + rc > 0 else 0.0
    return d


def per_kelas(r, nama_kelas):
    ada = list(getattr(r, "ap_class_index", []))  # kelas yang benar-benar ada di split ini
    out = {}
    for i, c in enumerate(ada):
        rec = {}
        for j, m in (("mask", r.seg), ("box", r.box)):
            try:
                p, rc, a50, a = m.class_result(i)
            except Exception:
                p = rc = a50 = float("nan")
                a = m.maps[c]
            rec.update({f"{j}_p": float(p), f"{j}_r": float(rc), f"{j}_map50": float(a50), f"{j}_map5095": float(a)})
        out[nama_kelas[c]] = rec
    return out


def jalankan(bobot, split_dir, args, out):
    from ultralytics import YOLO
    nama = f"{Path(bobot).stem}__{split_dir.name}"
    json_path = out / "run_json" / f"{nama}.json"
    if json_path.exists():
        return json.loads(json_path.read_text())
    yaml = str(split_dir / "data.yaml")
    t0 = time.time()
    model = YOLO(bobot)
    model.train(data=yaml, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, patience=args.patience,
                seed=args.seed_latih, deterministic=True, device=args.device, workers=args.workers,
                project=str(out / "runs"), name=nama, exist_ok=True, verbose=False)
    waktu = time.time() - t0
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    m = YOLO(str(best))
    params = sum(p.numel() for p in m.model.parameters())
    nama_kelas = m.names
    hasil = {"run": nama, "model": Path(bobot).stem, "split": split_dir.name,
             "rasio": int(re.search(r"_r(\d+)_", split_dir.name).group(1)),
             "seed_split": int(re.search(r"_s(\d+)$", split_dir.name).group(1)),
             "epoch_dilatih": args.epochs, "waktu_latih_menit": round(waktu / 60, 2),
             "parameter_juta": round(params / 1e6, 3), "ukuran_mb": round(best.stat().st_size / 1e6, 2)}
    for sp in ("val", "test"):
        if not (split_dir / "images" / sp).is_dir() or not any((split_dir / "images" / sp).iterdir()):
            continue
        r = m.val(data=yaml, split=sp, imgsz=args.imgsz, batch=args.batch, device=args.device,
                  plots=True, project=str(out / "plots"), name=f"{nama}_{sp}",
                  exist_ok=True, verbose=False)
        for k, v in ambil(r).items():
            hasil[f"{sp}_{k}"] = round(v, 5)
        if sp == "test":
            hasil["per_kelas_test"] = per_kelas(r, nama_kelas)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(hasil, indent=1))
    return hasil


def rata(v):
    v = [x for x in v if x is not None]
    return (statistics.fmean(v), statistics.pstdev(v) if len(v) > 1 else 0.0) if v else (float("nan"), 0.0)


def grafik_perbandingan(out, ringkas):
    """ringkas[(model, rasio)] = {metrik: (mean, std)} -> PNG di <out>/grafik/."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib tidak ada: grafik dilewati.")
        return
    d = Path(out) / "grafik"
    d.mkdir(parents=True, exist_ok=True)
    models = sorted({m for m, _ in ringkas})
    rasios = sorted({r for _, r in ringkas})
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.5))
    ax = axs[0]
    lebar = 0.8 / len(models)
    for i, m in enumerate(models):
        xs = [j - 0.4 + lebar * (i + 0.5) for j in range(len(rasios))]
        mu = [ringkas.get((m, r), {}).get("test_mask_map5095", (float("nan"), 0))[0] for r in rasios]
        sd = [ringkas.get((m, r), {}).get("test_mask_map5095", (0, 0))[1] for r in rasios]
        ax.bar(xs, mu, lebar, yerr=sd, capsize=3, label=m)
        for x, v in zip(xs, mu):
            if v == v:
                ax.text(x, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(rasios)))
    ax.set_xticklabels([f"rasio {r}" for r in rasios])
    ax.set_ylabel("test mask mAP50-95 (rata-rata +- sd antar seed)")
    ax.set_title("Perbandingan model x rasio split")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    ax = axs[1]
    ind = [("test_mask_p", "Presisi"), ("test_mask_r", "Recall"), ("test_mask_f1", "F1"),
           ("test_mask_map50", "mAP50"), ("test_mask_map5095", "mAP50-95")]
    kombinasi = sorted(ringkas, key=lambda x: (x[1], x[0]))
    lebar = 0.8 / len(ind)
    for i, (k, n) in enumerate(ind):
        xs = [j - 0.4 + lebar * (i + 0.5) for j in range(len(kombinasi))]
        ax.bar(xs, [ringkas[c][k][0] for c in kombinasi], lebar, label=n)
    ax.set_xticks(range(len(kombinasi)))
    ax.set_xticklabels([f"{m}\nr{r}" for m, r in kombinasi], fontsize=8)
    ax.set_ylim(0, 1.2)
    ax.set_title("Semua indikator mask (test, rata-rata antar seed)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=5, loc="upper center")
    fig.tight_layout()
    fig.savefig(d / "perbandingan_model_rasio.png", dpi=150)
    plt.close(fig)
    print(f"Grafik: {d / 'perbandingan_model_rasio.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", type=Path, default=ANN_DIR, help="folder berisi split_r*_s*/")
    ap.add_argument("--out", type=Path, default=(HASIL / "06_perbandingan_model") if HASIL else Path("hasil_06"))
    ap.add_argument("--models", nargs="+", default=["yolo11n-seg.pt", "yolo26n-seg.pt"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seed-latih", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--rasio", type=int, nargs="*", help="hanya rasio ini (mis. 80)")
    ap.add_argument("--seed-split", type=int, nargs="*", help="hanya seed split ini (mis. 44)")
    args = ap.parse_args()

    splits = sorted(p for p in Path(args.splits_dir).glob("split_r*_s*") if (p / "data.yaml").exists())
    if args.rasio:
        splits = [p for p in splits if int(re.search(r"_r(\d+)_", p.name).group(1)) in args.rasio]
    if args.seed_split:
        splits = [p for p in splits if int(re.search(r"_s(\d+)$", p.name).group(1)) in args.seed_split]
    if not splits:
        raise SystemExit("Tidak ada split_r*_s*/ ditemukan. Jalankan 05_siapkan_dataset_yolo.py dulu.")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(splits)} split x {len(args.models)} model = {len(splits) * len(args.models)} run.")

    semua = []
    for sp in splits:
        for bobot in args.models:
            print(f"\n{bobot} | {sp.name}")
            semua.append(jalankan(bobot, sp, args, args.out))

    kolom = ["run", "model", "split", "rasio", "seed_split", "epoch_dilatih", "waktu_latih_menit", "parameter_juta", "ukuran_mb"]
    kolom += sorted({k for r in semua for k in r if k not in kolom and k != "per_kelas_test"})
    with open(args.out / "hasil_run.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=kolom, extrasaction="ignore")
        w.writeheader()
        w.writerows(semua)
    with open(args.out / "per_kelas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        kc = [f"test_{j}_{m}" for j in ("mask", "box") for m in ("p", "r", "map50", "map5095")]
        w.writerow(["model", "split", "kelas"] + kc)
        for r in semua:
            for k, v in r.get("per_kelas_test", {}).items():
                if not isinstance(v, dict):  # cache lama: hanya mask mAP50-95
                    v = {"mask_map5095": v}
                w.writerow([r["model"], r["split"], k] + [f"{v[c[5:]]:.5f}" if c[5:] in v else "" for c in kc])

    # ---- ringkasan per (model, rasio): rata-rata lintas seed ----
    grup = defaultdict(list)
    for r in semua:
        grup[(r["model"], r["rasio"])].append(r)
    metrik = ["test_mask_map5095", "test_mask_map50", "test_mask_p", "test_mask_r", "test_mask_f1",
              "test_box_map5095", "test_box_map50", "test_box_p", "test_box_r", "test_box_f1",
              "val_mask_map5095", "val_mask_map50", "val_box_map5095", "test_ms_inferensi"]
    ringkas = {}
    with open(args.out / "ringkasan_rasio.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "rasio", "n_seed"] + [f"{m}_{s}" for m in metrik for s in ("mean", "std")] + ["selisih_val_test"])
        print(f"\n{'model':<16}{'rasio':>6}{'n':>3}{'test mask mAP50-95':>24}{'val mask mAP50-95':>20}{'val-test':>10}")
        for (model, rasio), rs in sorted(grup.items(), key=lambda x: (x[0][1], x[0][0])):
            baris, nilai = [model, rasio, len(rs)], {}
            for m in metrik:
                mu, sd = rata([r.get(m) for r in rs])
                nilai[m] = (mu, sd)
                baris += [f"{mu:.5f}", f"{sd:.5f}"]
            ringkas[(model, rasio)] = nilai
            gap = nilai["val_mask_map5095"][0] - nilai["test_mask_map5095"][0]
            w.writerow(baris + [f"{gap:.5f}"])
            tm, ts = nilai["test_mask_map5095"]
            vm, vs = nilai["val_mask_map5095"]
            print(f"{model:<16}{rasio:>6}{len(rs):>3}{tm:>17.4f}+-{ts:.4f}{vm:>13.4f}+-{vs:.4f}{gap:>10.4f}")
            print(f"{'':<25}mask  P {nilai['test_mask_p'][0]:.4f} R {nilai['test_mask_r'][0]:.4f} "
                  f"F1 {nilai['test_mask_f1'][0]:.4f} mAP50 {nilai['test_mask_map50'][0]:.4f} | "
                  f"kotak P {nilai['test_box_p'][0]:.4f} R {nilai['test_box_r'][0]:.4f} "
                  f"F1 {nilai['test_box_f1'][0]:.4f} mAP50 {nilai['test_box_map50'][0]:.4f} mAP50-95 {nilai['test_box_map5095'][0]:.4f}")

    if ringkas:
        grafik_perbandingan(args.out, ringkas)

    # ---- perbandingan berpasangan antar model pada split yang sama ----
    if len(args.models) == 2:
        a, b = (Path(m).stem for m in args.models)
        per_split = defaultdict(dict)
        for r in semua:
            per_split[r["split"]][r["model"]] = r
        selisih = [(s, d[b]["test_mask_map5095"] - d[a]["test_mask_map5095"]) for s, d in per_split.items()
                   if a in d and b in d and "test_mask_map5095" in d[a] and "test_mask_map5095" in d[b]]
        with open(args.out / "ringkasan_model.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["split", f"selisih_test_mask_map5095 ({b} - {a})"])
            for s, d in selisih:
                w.writerow([s, f"{d:.5f}"])
        if selisih:
            nilai = [d for _, d in selisih]
            mu, sd = rata(nilai)
            menang = sum(d > 0 for d in nilai)
            print(f"\n{b} vs {a}: selisih rata-rata test mask mAP50-95 = {mu:+.4f} (sd {sd:.4f}), "
                  f"{b} unggul di {menang}/{len(nilai)} split.")
            if len(nilai) < 5 or abs(mu) < sd:
                print("Selisih kecil dibanding sebarannya / split sedikit: jangan diklaim unggul. Tambah --seed di tahap 5.")
    print(f"\nHasil lengkap: {args.out}")


if __name__ == "__main__":
    main()