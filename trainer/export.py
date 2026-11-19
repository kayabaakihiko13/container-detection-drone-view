import argparse
import shutil
import sys
from pathlib import Path

import onnx
import onnxruntime as ort
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export model YOLO .pt ke ONNX (FP32, CPU, opset 11)."
    )
    parser.add_argument("--model", "-m", required=True, help="Path file model .pt")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Nama/path file .onnx. Jika hanya nama file, disimpan di folder yang sama dengan .pt.",
    )
    parser.add_argument(
        "--imgsz",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        default=[480, 480],
        help="Ukuran input ONNX [height width] (default: 480 480)",
    )
    parser.add_argument("--opset", type=int, default=11, help="Opset ONNX (default: 11)")
    parser.add_argument(
        "--simplify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Simplify graph ONNX (default: aktif)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device export; hanya 'cpu' yang didukung (default: cpu)",
    )
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Tidak didukung; export selalu FP32 (default: nonaktif)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Timpa file output jika sudah ada"
    )
    parser.add_argument(
        "--verify-infer",
        action="store_true",
        help="Jalankan inferensi dummy di onnxruntime setelah export",
    )
    return parser.parse_args()


def resolve_output(model_path: Path, output: str | None) -> Path:
    if output is None:
        out = model_path.with_suffix(".onnx")
    else:
        raw = Path(output).expanduser()
        out = raw.resolve() if len(raw.parts) > 1 else model_path.parent / raw.name

    if not out.suffix:
        out = out.with_suffix(".onnx")
    if out.suffix.lower() != ".onnx":
        raise ValueError(f"Output harus berekstensi .onnx, bukan '{out.suffix}'")
    return out


def verify_onnx(path: Path, run_infer: bool) -> None:
    proto = onnx.load(str(path))
    opset = next(
        (item.version for item in proto.opset_import if item.domain in {"", "ai.onnx"}),
        None,
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    print("=" * 60)
    print("Verifikasi ONNX")
    print("=" * 60)
    print(f"File        : {path}")
    print(f"Ukuran      : {path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"IR version  : {proto.ir_version}")
    print(f"Opset       : {opset}")
    print(f"Providers   : {session.get_providers()}")
    for inp in session.get_inputs():
        print(f"Input       : {inp.name} shape={inp.shape} dtype={inp.type}")
    for out in session.get_outputs():
        print(f"Output      : {out.name} shape={out.shape} dtype={out.type}")

    if run_infer:
        import numpy as np

        feeds = {}
        for inp in session.get_inputs():
            dtype = np.float16 if inp.type == "tensor(float16)" else np.float32
            shape = [dim if isinstance(dim, int) else 1 for dim in inp.shape]
            feeds[inp.name] = np.zeros(shape, dtype=dtype)

        outputs = session.run(None, feeds)
        print("Dummy inference: OK")
        for out, value in zip(session.get_outputs(), outputs):
            print(f"  {out.name}: {value.shape}")

    print("=" * 60)


def main():
    args = parse_args()

    if args.device.lower() != "cpu":
        print("ERROR: script ini hanya mendukung --device cpu.", file=sys.stderr)
        return 1
    if args.half:
        print("ERROR: --half tidak didukung; export selalu FP32 untuk CPU.", file=sys.stderr)
        return 1
    if any(size <= 0 for size in args.imgsz):
        print("ERROR: --imgsz HEIGHT WIDTH harus bilangan bulat positif.", file=sys.stderr)
        return 1
    if args.opset <= 0:
        print("ERROR: --opset harus bilangan bulat positif.", file=sys.stderr)
        return 1

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        print(f"ERROR: file model tidak ditemukan: {model_path}", file=sys.stderr)
        return 1
    if model_path.suffix.lower() != ".pt":
        print(f"ERROR: model harus berformat .pt: {model_path}", file=sys.stderr)
        return 1

    try:
        output_path = resolve_output(model_path, args.output)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if output_path.exists() and output_path.is_dir():
        print(f"ERROR: output berupa folder: {output_path}", file=sys.stderr)
        return 1

    auto_path = model_path.with_suffix(".onnx")
    conflicts = [path for path in {output_path, auto_path} if path.exists()]
    if conflicts and not args.overwrite:
        for path in conflicts:
            print(
                f"ERROR: file sudah ada: {path} (gunakan --overwrite)",
                file=sys.stderr,
            )
        return 1

    height, width = args.imgsz

    print("=" * 60)
    print("YOLO Export ONNX")
    print("=" * 60)
    print(f"Model        : {model_path}")
    print(f"Output       : {output_path}")
    print(f"Input size   : {height} x {width} (H x W)")
    print("Presisi      : FP32")
    print(f"Opset        : {args.opset}")
    print(f"Simplify     : {args.simplify}")
    print("Device       : cpu")
    print("=" * 60)

    model = YOLO(str(model_path))
    exported = Path(
        model.export(
            format="onnx",
            imgsz=[height, width],
            simplify=args.simplify,
            opset=args.opset,
            device="cpu",
        )
    )

    if exported.resolve() != output_path.resolve():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()
        shutil.move(str(exported), str(output_path))
        exported = output_path

    print()
    try:
        verify_onnx(exported, args.verify_infer)
    except Exception as exc:
        print(f"WARNING: verifikasi ONNX gagal: {exc}", file=sys.stderr)

    print(f"Export selesai: {exported}")
    print(
        f'Prediksi      : yolo predict model="{exported}" '
        f'source="path/ke/gambar" imgsz={height}'
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())