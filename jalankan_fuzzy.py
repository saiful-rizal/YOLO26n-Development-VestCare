import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from fuzzysugeno import analisis_hasil


def sumber_video(value):
    return int(value) if value.isdigit() else value


def main():
    ap = argparse.ArgumentParser(description="YOLO-seg + Fuzzy Sugeno")
    ap.add_argument("--model", type=Path, default=Path("models") / "best.pt")
    ap.add_argument("--source", default="0", help="0 untuk webcam, atau path gambar/video")
    ap.add_argument("--conf", type=float, default=0.60,
                    help="confidence minimum YOLO sebelum diteruskan ke fuzzy")
    ap.add_argument("--jarak", type=float, default=2.0,
                    help="jarak sementara dalam meter; ganti dengan sensor/estimasi kamera")
    args = ap.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model tidak ditemukan: {args.model}")

    model = YOLO(str(args.model))
    kelas_model = set(model.names.values())
    kelas_fuzzy = {"pohon", "lubang", "orang", "motor", "mobil"}
    kelas_tidak_didukung = kelas_model - kelas_fuzzy
    if kelas_tidak_didukung:
        raise SystemExit(
            "Kelas model tidak sesuai dataset fuzzy: "
            + ", ".join(sorted(kelas_tidak_didukung))
        )
    source = sumber_video(args.source)
    hasil = model.predict(source=source, imgsz=416, conf=args.conf, stream=True, verbose=False)

    for res in hasil:
        analisis = analisis_hasil(
            res,
            jarak_fn=lambda box, nama: args.jarak,
            conf_min=args.conf,
        )
        frame = res.plot()
        if not analisis:
            cv2.putText(frame, "Tidak ada deteksi yang memenuhi confidence",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (180, 180, 180), 2, cv2.LINE_AA)
        for i, item in enumerate(analisis):
            teks = (f"{item['kelas']} {item['conf']:.2f} -> {item['level']} "
                    f"{item['urgensi']:.1f} | {item['jarak_m']:.1f} m")
            cv2.putText(frame, teks, (10, 30 + i * 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 255) if item["urgensi"] >= 75 else (0, 200, 255),
                        2, cv2.LINE_AA)
            print(item["pesan"])
        cv2.imshow("YOLO + Fuzzy Sugeno", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
