# Drone Container Detection

Proyek deteksi kontainer dari citra drone menggunakan YOLO (Ultralytics), object tracking,
panorama stitching ALIKED + LightGlue, dan export model ONNX untuk deployment CPU.

## Lingkungan

- Python 3.11 (disarankan)
- Windows / PowerShell
- Default instalasi memakai PyTorch CPU
- GPU NVIDIA opsional (lihat bagian [GPU opsional](#gpu-opsional))

## Instalasi

Jalankan dari root proyek `D:\Dataset\Kerja\Drone`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

`requirements-dev.txt` memuat seluruh dependency untuk:

- training (`trainer/train.py`)
- pipeline panorama (`pipeline_yolo_paronama.py`)
- export ONNX (`trainer/export.py`)

LightGlue dipasang dari repository resmi pada commit tertentu, sehingga proses instalasi
membutuhkan koneksi internet dan `git`. Untuk instalasi offline, gunakan clone lokal:

```powershell
pip install -e D:\Tools\LightGlue
```

### GPU opsional

Requirement default memakai PyTorch CPU (`torch.cuda.is_available()` bernilai `False`).
Jika ingin training/inferensi di GPU, install ulang PyTorch dengan build CUDA:

```powershell
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
```

Sesuaikan `cu128` dengan versi CUDA driver yang terpasang.

### Verifikasi instalasi

```powershell
python -c "import torch, ultralytics, cv2, numpy, lap; print('core ok', torch.__version__, cv2.__version__)"
python -c "import lightglue, kornia, tqdm, matplotlib; print('panorama ok')"
python -c "import onnx, onnxruntime, onnxslim; print('export ok')"
python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

### Membersihkan environment lama

Jika sebelumnya sudah ada paket versi lain, rapikan agar tidak ada distribusi ganda
(terutama OpenCV dan ONNX Runtime):

```powershell
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless
pip uninstall -y onnxruntime onnxruntime-gpu
pip uninstall -y lightglue
```

Distribusi yang dipakai proyek ini hanya `opencv-python` dan `onnxruntime` (CPU).

## Struktur proyek

```
.
├── dataset.yaml                     # dataset root (urutan kelas berbeda)
├── YoloDataset/dataset.yaml         # dataset yang dipakai training labelme_yolo-2
├── trainer/
│   ├── train.py                     # training YOLO
│   ├── export.py                    # export .pt -> .onnx (FP32, CPU, opset 11)
│   ├── yaml/hyp.yaml                # hyperparameter & augmentasi
│   ├── custom_bytetrack.yaml        # konfigurasi ByteTrack
│   └── runs/detect/                 # output training, weights, metrik
├── pipeline_yolo_paronama.py        # tracking + panorama ALIKED/LightGlue
├── pipeline_hybrid_trackless.py     # pipeline panorama tanpa tracker
├── track.py                         # tracking video ke MP4 720p
├── split_video.py                   # pemecah video menjadi frame
├── split_panorama.py                # pemecah panorama
├── filter_track_ids.py              # filter hasil track JSONL
└── requirements-dev.txt
```

## Dataset

Gunakan `YoloDataset/dataset.yaml` untuk training dan validasi. Urutan kelasnya:

```
0: container_left_right
1: iso_code
2: container_number
3: container_rear_front
4: container_top
```

Struktur folder:

```
YoloDataset/
├── dataset.yaml
├── images/train
├── images/val
├── labels/train
└── labels/val
```

Perhatian: `dataset.yaml` di root memakai urutan kelas yang berbeda. Jangan mencampur
keduanya, karena label akan terbaca dengan kelas yang salah.

## Penggunaan

### Training

```powershell
python trainer\train.py `
  --data "YoloDataset\dataset.yaml" `
  --model "yolov8m.pt" `
  --cfg "trainer\yaml\hyp.yaml" `
  --epochs 50 `
  --imgsz 480 `
  --batch 8 `
  --device cpu `
  --project "trainer\runs\detect" `
  --name "labelme_yolo"
```

- `--device 0` untuk GPU (jika PyTorch CUDA terpasang), `--device cpu` untuk CPU.
- `--cfg` sebaiknya diisi eksplisit karena default `trainer/hyp.yaml` tidak ada;
  file yang tersedia adalah `trainer/yaml/hyp.yaml`.
- Output training: `trainer/runs/detect/<name>/` berisi `weights/best.pt`,
  `results.csv`, grafik kurva, dan confusion matrix.

### Validasi model

```powershell
yolo detect val `
  model="trainer\runs\detect\labelme_yolo-2\weights\best.pt" `
  data="YoloDataset\dataset.yaml" `
  imgsz=480 batch=8 device=cpu `
  project="trainer\runs\detect" name="val_best" exist_ok=True
```

Metrik utama: `metrics/mAP50(B)`, `metrics/mAP50-95(B)`, precision, dan recall.
Metrik juga tersimpan di `trainer/runs/detect/<name>/results.csv`.

### Export ONNX

```powershell
python trainer\export.py `
  -m "trainer\runs\detect\labelme_yolo-2\weights\best.pt" `
  -o "container_detection_model.onnx" `
  --imgsz 480 480 `
  --opset 11
```

Karakteristik `trainer/export.py`:

- Input wajib `.pt`, output `.onnx`
- Presisi selalu FP32 (`--half` ditolak)
- Device selalu CPU (`--device cpu`)
- Opset default `11`
- `--simplify` aktif secara default; gunakan `--no-simplify` untuk mematikannya
- `--imgsz HEIGHT WIDTH` mendefinisikan bentuk input ONNX `[1, 3, H, W]`
- Output default disimpan di folder yang sama dengan file `.pt`

Contoh lain:

```powershell
# verifikasi onnxruntime + dummy inference
python trainer\export.py -m "...\best.pt" -o "model.onnx" --verify-infer

# timpa file output yang sudah ada
python trainer\export.py -m "...\best.pt" -o "model.onnx" --overwrite

# ukuran non-persegi (height 480, width 640)
python trainer\export.py -m "...\best.pt" -o "model_480x640.onnx" --imgsz 480 640
```

### Tracking video

```powershell
python track.py `
  --source "DJI_0022.MP4" `
  --model "trainer\runs\detect\labelme_yolo-2\weights\best.pt" `
  --tracker "trainer\custom_bytetrack.yaml" `
  --imgsz 480 `
  --conf 0.3 `
  --device cpu `
  --output-dir "runs\track\DJI_0022_720p"
```

Output: video MP4 720p, `tracks.jsonl`, dan file pendukung di `--output-dir`.
Jika `--device cpu`, inferensi lebih lambat dibanding GPU.

### Pipeline panorama

```powershell
python pipeline_yolo_paronama.py `
  --source "DJI_0022.MP4" `
  --model "trainer\runs\detect\labelme_yolo-2\weights\best.pt" `
  --profile balanced `
  --tracker bytetrack.yaml `
  --device cpu `
  --classes 4
```

- `--profile`: `hq`, `balanced`, atau `fast`.
- `--classes 4` memilih kelas `container_top` (lihat urutan kelas `YoloDataset`).
- `--classes 0` memilih `container_left_right`.
- Output default: `runs/lightglue_tracking/`, berisi panorama, crop, analytics CSV,
  `summary_report.json`, dan grafik (jika matplotlib tersedia).
- Gunakan `--save-crops` dan `--debug-matches` untuk inspeksi hasil matching.

## Catatan

- Model `labelme_yolo-2` dilatih dengan `YoloDataset/dataset.yaml` pada `imgsz=480`.
  Gunakan resolusi dan urutan kelas yang sama saat training, validasi, dan inferensi.
- `onnxruntime` versi CPU sudah cukup untuk `trainer/export.py`.
- `lap` wajib terpasang agar ByteTrack berjalan.
- LightGlue membutuhkan `torch`, `torchvision`, `kornia`, dan OpenCV.
- File `test.py` kosong dan folder `tests/` masih merujuk modul yang tidak ada di repo.

## Troubleshooting

**`ModuleNotFoundError: lightglue`**
```powershell
pip install -r requirements-dev.txt
# atau
pip install -e D:\Tools\LightGlue
```

**Konflik OpenCV / `cv2` error**
```powershell
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless
pip install opencv-python==4.10.0.84
```

**`torch.cuda.is_available()` bernilai False**
Itu normal bila memakai requirement default (CPU). Untuk GPU:
```powershell
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
```

**Export gagal karena file output sudah ada**
```powershell
python trainer\export.py -m "...\best.pt" -o "model.onnx" --overwrite
```
