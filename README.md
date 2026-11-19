# Image Stitching App — Drone Container Detection

Proyek deteksi kontainer dari video drone menggunakan YOLO (Ultralytics) + ByteTrack,
lalu menyusun crop per `track_id` menjadi panorama dengan ALIKED + LightGlue.
Tersedia juga script training, export model ke ONNX (CPU/FP32), dan konfigurasi
hyperparameter/tracker khusus citra drone.

## Lingkungan

- Python 3.11 (disarankan)
- Windows / PowerShell
- CPU-only: aplikasi berjalan sepenuhnya di CPU tanpa GPU

## Instalasi

Jalankan dari root proyek `D:\Kerja\image_stitching_app`.

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

### Verifikasi instalasi

```powershell
python -c "import torch, ultralytics, cv2, numpy, lap; print('core ok', torch.__version__, cv2.__version__)"
python -c "import lightglue, kornia, tqdm, matplotlib; print('panorama ok')"
python -c "import onnx, onnxruntime, onnxslim; print('export ok')"
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
├── pipeline_yolo_paronama.py            # tracking YOLO + ByteTrack -> panorama ALIKED/LightGlue
├── requirements-dev.txt
├── hyper_config/
│   ├── container_detection_hyp.yaml     # hyperparameter & augmentasi training (cfg)
│   └── custom_bytetrack.yaml            # konfigurasi ByteTrack untuk citra drone
├── trainer/
│   ├── train.py                         # training YOLO
│   └── export.py                        # export .pt -> .onnx (FP32, CPU, opset 11)
└── weight/
    ├── best.pt                          # model YOLO hasil training
    └── container_detection_model.onnx   # model hasil export ONNX
```

Dataset, video input, dan output run tidak disertakan di repository.

## Dataset

Script training menerima path `dataset.yaml` lewat argumen `--data`. Default internal
script menunjuk ke `..\container-top-yolo\dataset.yaml` yang tidak disertakan di repo,
jadi selalu isi `--data` secara eksplisit.

Struktur dataset yang diharapkan Ultralytics:

```
dataset/
├── dataset.yaml
├── images/train
├── images/val
├── labels/train
└── labels/val
```

Contoh `dataset.yaml` (sesuaikan `path` dan `names` dengan anotasi):

```yaml
path: D:/path/ke/dataset
train: images/train
val: images/val

names:
  0: container_top
```

Urutan `names` harus sama dengan urutan kelas pada label dan dengan nilai `--classes`
saat menjalankan pipeline panorama.

## Penggunaan

### Training

```powershell
python trainer\train.py `
  --data "D:\path\ke\dataset\dataset.yaml" `
  --cfg "hyper_config\container_detection_hyp.yaml" `
  --model "yolov8n.pt" `
  --epochs 50 `
  --imgsz 640 `
  --batch 8 `
  --device cpu `
  --project "trainer\runs\detect" `
  --name "container_top_exp"
```

- `--device cpu`: aplikasi ini hanya berjalan di CPU.
- `--cfg` default mengarah ke `trainer/hyp.yaml` yang tidak ada di repo, jadi isi
  eksplisit dengan `hyper_config\container_detection_hyp.yaml`. Jika `--cfg` tidak
  ditemukan, training memakai default Ultralytics.
- `--model` default `yolov8n.pt`; bisa diganti `yolov8s.pt`, `yolo11n.pt`, dan sebagainya.
- `--resume` untuk melanjutkan training yang terputus.
- Script otomatis menjalankan `model.val()` setelah training.
- Output: `<project>\<name>\` berisi `weights/best.pt`, `weights/last.pt`, `results.csv`,
  grafik kurva, dan confusion matrix.

### Export ONNX

```powershell
python trainer\export.py `
  -m "weight\best.pt" `
  -o "weight\container_detection_model.onnx" `
  --imgsz 480 480 `
  --opset 11 `
  --overwrite
```

`--overwrite` diperlukan bila file output sudah ada (seperti model ONNX yang sudah
disertakan di folder `weight/`).

Karakteristik `trainer/export.py`:

- Input wajib `.pt`, output wajib `.onnx`
- Presisi selalu FP32 (`--half` ditolak)
- Device selalu CPU (`--device cpu`)
- Opset default `11`
- `--simplify` aktif secara default; gunakan `--no-simplify` untuk mematikannya
- `--imgsz HEIGHT WIDTH` mendefinisikan bentuk input ONNX `[1, 3, H, W]`
- Jika `-o` hanya berupa nama file, output disimpan di folder yang sama dengan `.pt`
- File output yang sudah ada akan ditolak kecuali memakai `--overwrite`

Contoh lain:

```powershell
# verifikasi onnxruntime + dummy inference
python trainer\export.py -m "weight\best.pt" -o "weight\model.onnx" --verify-infer

# timpa file output yang sudah ada
python trainer\export.py -m "weight\best.pt" -o "weight\model.onnx" --overwrite

# ukuran non-persegi (height 480, width 640)
python trainer\export.py -m "weight\best.pt" -o "weight\model_480x640.onnx" --imgsz 480 640
```

### Pipeline panorama

```powershell
python pipeline_yolo_paronama.py `
  --source "DJI_0022.MP4" `
  --model "weight\best.pt" `
  --tracker "hyper_config\custom_bytetrack.yaml" `
  --profile balanced `
  --device cpu `
  --classes 0
```

Alur pemrosesan:

1. YOLO + ByteTrack mendeteksi dan mengelompokkan objek berdasarkan `track_id`.
2. Crop tiap track disimpan berurutan sesuai frame video (dibatasi reservoir sampling).
3. ALIKED mengekstrak keypoint, LightGlue mencocokkan crop berurutan.
4. Transformasi divalidasi RANSAC affine/homografi; pasangan yang gagal ditolak
   (tidak ada fallback pergeseran).
5. Setiap track dengan cukup crop valid dijahit menjadi satu panorama.

Argumen penting:

- `--profile`: `hq`, `balanced` (default), atau `fast`.
- `--tracker`: default `bytetrack.yaml` bawaan Ultralytics; gunakan
  `hyper_config\custom_bytetrack.yaml` untuk konfigurasi drone.
- `--classes`: class ID YOLO dipisahkan koma (mis. `0` atau `0,3`), sesuai `names`
  pada dataset.
- `--conf`: ambang confidence deteksi (default `0.40`).
- `--device`: gunakan `cpu`; aplikasi ini hanya berjalan di CPU.
- `--save-crops`: simpan crop per track ke `<output-dir>\_crops`.
- `--debug-matches`: simpan visual pasangan crop ke `<output-dir>\_debug_matches`.
- `--seed`: seed acak (default `42`).

Parameter yang bisa ditimpa manual: `--imgsz`, `--vid-stride`, `--max-crops`,
`--min-track-len`, `--crop-pad`, `--max-crop-dim`, `--min-crop-size`,
`--min-crop-blur`, `--max-stitch-frames`, `--keypoints`, `--min-inliers`,
`--min-inlier-ratio`, `--ransac-thresh`, `--max-pano-mp`, `--jpeg-quality`.

Profil bawaan:

| Profil     | imgsz | vid_stride | max_crops | max_stitch_frames | keypoints | min_track_len |
|------------|-------|------------|-----------|-------------------|-----------|---------------|
| `hq`       | 1024  | 1          | 300       | 90                | 6144      | 10            |
| `balanced` | 800   | 2          | 160       | 40                | 3072      | 8             |
| `fast`     | 640   | 3          | 100       | 30                | 2048      | 6             |

Output default di `runs/lightglue_tracking/`:

- `container_<track_id>.jpg` — panorama per track
- `frame_analytics.csv`, `track_analytics.csv`, `matching_analytics.csv`,
  `panorama_analytics.csv`, `stage_timing.csv`, `errors.csv`
- `summary_report.json` — ringkasan parameter, tracking, matching, stitching, sistem
- `analytics_charts/` — grafik PNG (jika matplotlib tersedia)
- `_crops/` dan `_debug_matches/` hanya jika argumen terkait diaktifkan

## Konfigurasi tambahan

### `hyper_config/container_detection_hyp.yaml`

Hyperparameter dan augmentasi untuk citra drone/tampak atas:

- Optimizer `auto`, `lr0=0.01`, cosine LR, warmup 3 epoch
- Rotasi `degrees=25`, `scale=0.5`, `shear=2.0`, `perspective=0.0002`
- Flip horizontal dan vertikal masing-masing 50%
- Mosaic 100% (dimatikan 10 epoch terakhir), mixup 10%, erasing 20%

### `hyper_config/custom_bytetrack.yaml`

ByteTrack yang lebih toleran terhadap gerakan kamera drone:

- `track_high_thresh=0.2`, `track_low_thresh=0.05`, `new_track_thresh=0.2`
- `track_buffer=60`, `match_thresh=0.75`, `fuse_score=True`

## Catatan

- Gunakan resolusi dan urutan kelas yang sama saat training, export, dan inferensi.
- Model contoh tersedia di `weight/best.pt` dan `weight/container_detection_model.onnx`.
- `onnxruntime` versi CPU sudah cukup untuk `trainer/export.py`.
- `lap` wajib terpasang agar ByteTrack berjalan.
- LightGlue membutuhkan `torch`, `torchvision`, `kornia`, dan OpenCV.
- `track_id` yang salah menggabungkan dua kontainer dapat menghasilkan panorama
  campuran; periksa `_crops` dan `_debug_matches` untuk memvalidasi tracking.

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

**Dataset tidak ditemukan saat training**
Default `--data` mengarah ke folder yang tidak ada di repo. Isi eksplisit:
```powershell
python trainer\train.py --data "D:\path\ke\dataset\dataset.yaml"
```

**Export gagal karena file output sudah ada**
```powershell
python trainer\export.py -m "weight\best.pt" -o "weight\model.onnx" --overwrite
```

**Panorama tidak terbentuk / track dilewati**
Naikkan `--max-crops`/`--min-track-len` atau gunakan `--profile hq`, lalu periksa
`errors.csv`, `matching_analytics.csv`, dan `_debug_matches` untuk melihat alasan
penolakan pasangan crop.
