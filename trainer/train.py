#!/usr/bin/env python3
"""
Script pelatihan YOLO untuk deteksi container_top.
Mendukung Ultralytics YOLO (YOLOv8 / YOLO11) dengan akselerasi GPU CUDA,
serta mendukung custom hyperparameters dan data augmentasi via file YAML (parameter cfg).
"""

import argparse
import os
import sys
from pathlib import Path
import torch
from ultralytics import YOLO


def parse_args():
    # Direktori dasar script trainer
    base_dir = Path(__file__).resolve().parent
    default_data = base_dir.parent / "container-top-yolo" / "dataset.yaml"
    default_cfg = base_dir / "hyp.yaml"
    default_project = base_dir / "runs" / "detect"

    parser = argparse.ArgumentParser(
        description="Train YOLO Model untuk deteksi container_top."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(default_data),
        help=f"Path ke file dataset.yaml (default: {default_data})"
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default=str(default_cfg),
        help=f"Path ke file konfigurasi hyperparameter & augmentasi (default: {default_cfg})"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Pretrained model weights (contoh: yolov8n.pt, yolov8s.pt, yolo11n.pt). Default: yolov8n.pt"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Jumlah epoch pelatihan (default: 50)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Ukuran resolusi gambar input training (default: 640)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size (default: 8 untuk GPU 6GB)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0" if torch.cuda.is_available() else "cpu",
        help="Device untuk training: '0', '0,1', atau 'cpu' (default: 0 jika GPU tersedia)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Jumlah worker dataloader (default: 2 untuk Windows)"
    )
    parser.add_argument(
        "--project",
        type=str,
        default=str(default_project),
        help=f"Folder output hasil training (default: {default_project})"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="container_top_exp",
        help="Nama eksperimen / subfolder run (default: container_top_exp)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Lanjutkan pelatihan dari checkpoint terakhir jika terputus"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validasi path dataset.yaml
    data_path = Path(args.data).resolve()
    if not data_path.exists():
        print(f"ERROR: File dataset tidak ditemukan di: {data_path}", file=sys.stderr)
        return 1

    # Validasi path cfg (hyperparameter / augmentasi)
    cfg_path = Path(args.cfg).resolve() if args.cfg else None
    has_cfg = cfg_path and cfg_path.exists()

    print("=" * 60)
    print("🚀 YOLO Training Setup")
    print("=" * 60)
    print(f"Dataset Config : {data_path}")
    print(f"Hyp & Augment  : {cfg_path if has_cfg else 'Default Ultralytics'}")
    print(f"Base Model     : {args.model}")
    print(f"Epochs         : {args.epochs}")
    print(f"Batch Size     : {args.batch}")
    print(f"Image Size     : {args.imgsz}")
    print(f"Device         : {args.device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() and args.device != 'cpu' else 'CPU'})")
    print(f"Output Project : {args.project}")
    print(f"Run Name       : {args.name}")
    print("=" * 60)

    # Inisialisasi model
    model = YOLO(args.model)

    # Siapkan argumen training
    train_args = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "resume": args.resume,
        "plots": True,
        "save": True
    }

    # Jika file cfg ada, masukkan ke parameter train
    if has_cfg:
        train_args["cfg"] = str(cfg_path)

    # Mulai training
    print("\n⏳ Memulai proses pelatihan...\n")
    results = model.train(**train_args)

    # Jalankan validasi akhir
    print("\n📊 Menjalankan validasi akhir dengan model terbaik...")
    metrics = model.val()

    # Lokasi weights terbaik
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print("✅ Training Selesai!")
    if best_weights.exists():
        print(f"Model Terbaik Tersimpan di: {best_weights}")
    print(f"Log & Grafik Training      : {Path(args.project) / args.name}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
