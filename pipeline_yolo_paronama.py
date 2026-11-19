import argparse
import csv
import gc
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from lightglue import ALIKED, LightGlue

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x


PROFILES = {
    "hq": {
        "imgsz": 1024, "vid_stride": 1, "max_crops": 300,
        "crop_pad": 0.04, "max_crop_dim": 1280,
        "max_stitch_frames": 90, "keypoints": 6144,
        "min_inliers": 30, "min_inlier_ratio": 0.40,
        "ransac_thresh": 2.0, "min_track_len": 10,
        "min_crop_size": 48, "min_crop_blur": 10.0,
        "max_pano_mp": 60.0, "jpeg_quality": 98,
    },
    "balanced": {
        "imgsz": 800, "vid_stride": 2, "max_crops": 160,
        "crop_pad": 0.06, "max_crop_dim": 768,
        "max_stitch_frames": 40, "keypoints": 3072,
        "min_inliers": 22, "min_inlier_ratio": 0.32,
        "ransac_thresh": 2.2, "min_track_len": 8,
        "min_crop_size": 40, "min_crop_blur": 10.0,
        "max_pano_mp": 35.0, "jpeg_quality": 96,
    },
    "fast": {
        "imgsz": 640, "vid_stride": 3, "max_crops": 100,
        "crop_pad": 0.05, "max_crop_dim": 640,
        "max_stitch_frames": 30, "keypoints": 2048,
        "min_inliers": 18, "min_inlier_ratio": 0.28,
        "ransac_thresh": 2.8, "min_track_len": 6,
        "min_crop_size": 32, "min_crop_blur": 8.0,
        "max_pano_mp": 25.0, "jpeg_quality": 94,
    },
}



def write_csv(path, rows, fieldnames=None):
    """Simpan list of dict ke CSV; aman untuk hasil kosong."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        if not fieldnames:
            handle.write("\n")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_mean(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def safe_median(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else None


def save_analytics_charts(out_dir, frame_rows, track_rows, match_rows, stage_rows):
    """Buat grafik PNG terpisah; matplotlib bersifat opsional."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        print(f"[WARN] Grafik tidak dibuat (matplotlib tidak tersedia): {error}")
        return

    chart_dir = Path(out_dir) / "analytics_charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    def finish(fig, filename):
        fig.tight_layout()
        fig.savefig(chart_dir / filename, dpi=160, bbox_inches="tight")
        plt.close(fig)

    if frame_rows:
        x = [r["processed_frame"] for r in frame_rows]
        y = [r["tracked_detections"] for r in frame_rows]
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(x, y, linewidth=1.2)
        ax.set(title="Tracked detections per processed frame", xlabel="Processed frame", ylabel="Tracked detections")
        ax.grid(True, alpha=0.25)
        finish(fig, "01_detections_per_frame.png")

        conf = [r["mean_confidence"] for r in frame_rows if r["mean_confidence"] is not None]
        if conf:
            fig, ax = plt.subplots(figsize=(11, 4.5))
            ax.plot(range(1, len(conf) + 1), conf, linewidth=1.2)
            ax.set(title="Mean YOLO confidence per frame", xlabel="Processed frame with detections", ylabel="Mean confidence", ylim=(0, 1))
            ax.grid(True, alpha=0.25)
            finish(fig, "02_confidence_per_frame.png")

    if track_rows:
        labels = [str(r["track_id"]) for r in track_rows]
        continuity = [r["continuity_proxy"] for r in track_rows]
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.45), 4.5))
        ax.bar(labels, continuity)
        ax.set(title="Track temporal continuity (proxy)", xlabel="Track ID", ylabel="Continuity ratio", ylim=(0, 1.05))
        ax.grid(True, axis="y", alpha=0.25)
        finish(fig, "03_track_continuity.png")

    if match_rows:
        valid = [r for r in match_rows if r.get("inlier_ratio") is not None]
        if valid:
            fig, ax = plt.subplots(figsize=(11, 4.5))
            ax.plot(range(1, len(valid) + 1), [r["inlier_ratio"] for r in valid], marker="o", markersize=3, linewidth=1)
            ax.set(title="ALIKED + LightGlue RANSAC inlier ratio", xlabel="Pair attempt", ylabel="Inlier ratio", ylim=(0, 1.05))
            ax.grid(True, alpha=0.25)
            finish(fig, "04_matching_inlier_ratio.png")

        errors = [r["mean_inlier_reprojection_error_px"] for r in match_rows if r.get("mean_inlier_reprojection_error_px") is not None]
        if errors:
            fig, ax = plt.subplots(figsize=(11, 4.5))
            ax.plot(range(1, len(errors) + 1), errors, marker="o", markersize=3, linewidth=1)
            ax.set(title="RANSAC inlier reprojection residual", xlabel="Pair with valid transform", ylabel="Mean residual (pixels)")
            ax.grid(True, alpha=0.25)
            finish(fig, "05_reprojection_residual.png")

        fig, ax = plt.subplots(figsize=(7, 4.5))
        accepted = sum(bool(r.get("accepted")) for r in match_rows)
        rejected = len(match_rows) - accepted
        ax.bar(["Accepted", "Rejected"], [accepted, rejected])
        ax.set(title="Pair registration outcomes", ylabel="Number of pairs")
        ax.grid(True, axis="y", alpha=0.25)
        finish(fig, "06_registration_outcomes.png")

    if stage_rows:
        names = [r["stage"] for r in stage_rows]
        seconds = [r["seconds"] for r in stage_rows]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(names, seconds)
        ax.set(title="Pipeline stage processing time", ylabel="Seconds")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, axis="y", alpha=0.25)
        finish(fig, "07_stage_runtime.png")

    print(f"[ANALYTICS] Grafik disimpan: {chart_dir}")


def export_analytics(out_dir, summary, frame_rows, track_rows, match_rows, panorama_rows, error_rows, stage_rows):
    out_dir = Path(out_dir)
    write_csv(out_dir / "frame_analytics.csv", frame_rows)
    write_csv(out_dir / "track_analytics.csv", track_rows)
    write_csv(out_dir / "matching_analytics.csv", match_rows)
    write_csv(out_dir / "panorama_analytics.csv", panorama_rows)
    write_csv(out_dir / "errors.csv", error_rows, ["stage", "track_id", "pair_index", "error_type", "message"])
    write_csv(out_dir / "stage_timing.csv", stage_rows)
    with (out_dir / "summary_report.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    save_analytics_charts(out_dir, frame_rows, track_rows, match_rows, stage_rows)
    print("[ANALYTICS] summary_report.json, CSV, errors.csv, dan grafik selesai dibuat.")

def parse_args():
    p = argparse.ArgumentParser(
        description="YOLO Tracking + ALIKED/LightGlue Container Panorama"
    )
    p.add_argument("--source", required=True)
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--output-dir", default="runs/lightglue_tracking")
    p.add_argument("--profile", choices=PROFILES, default="balanced")

    p.add_argument("--conf", type=float, default=0.40)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--vid-stride", type=int, default=None)
    p.add_argument("--max-crops", type=int, default=None)
    p.add_argument("--min-crops-per-track", type=int, default=25,
                   help="Track dengan crop kurang dari nilai ini dilewati; default 25.")
    p.add_argument("--min-track-len", type=int, default=None)

    p.add_argument("--crop-pad", type=float, default=None)
    p.add_argument("--max-crop-dim", type=int, default=None)
    p.add_argument("--min-crop-size", type=int, default=None)
    p.add_argument("--min-crop-blur", type=float, default=None)

    p.add_argument("--max-stitch-frames", type=int, default=None)
    p.add_argument("--keypoints", type=int, default=None)
    p.add_argument("--min-inliers", type=int, default=None)
    p.add_argument("--min-inlier-ratio", type=float, default=None)
    p.add_argument("--ransac-thresh", type=float, default=None)
    p.add_argument("--max-pano-mp", type=float, default=None)
    p.add_argument("--jpeg-quality", type=int, default=None)

    p.add_argument("--classes", type=str, default=None,
                   help="Class ID YOLO dipisahkan koma, contoh: 0")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-crops", action="store_true")
    p.add_argument("--debug-matches", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def apply_profile(args):
    preset = PROFILES[args.profile]
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def resize_longest(image, max_dim):
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image
    scale = max_dim / float(longest)
    return cv2.resize(
        image,
        (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def make_crop(frame, box, pad_ratio, max_dim, return_bounds=False):
    """Crop YOLO box and optionally return crop bounds in original video pixels."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 1 or bh <= 1:
        return (None, None) if return_bounds else None

    px = int(round(bw * pad_ratio))
    py = int(round(bh * pad_ratio))
    xa, ya = max(0, x1 - px), max(0, y1 - py)
    xb, yb = min(w, x2 + px), min(h, y2 + py)
    if xb <= xa or yb <= ya:
        return (None, None) if return_bounds else None

    crop = resize_longest(frame[ya:yb, xa:xb].copy(), max_dim)
    bounds = {
        "x": int(xa),
        "y": int(ya),
        "width": int(xb - xa),
        "height": int(yb - ya),
    }
    return (crop, bounds) if return_bounds else crop


def valid_crop(image, min_size, min_blur):
    if image is None or image.size == 0:
        return False
    h, w = image.shape[:2]
    if min(h, w) < min_size:
        return False
    if min_blur > 0:
        small = resize_longest(image, 512)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_32F).var() < min_blur:
            return False
    return True


def choose_ordered_samples(entries, limit):
    """Pilih sampel merata dari urutan waktu, bukan reservoir acak."""
    if len(entries) <= limit:
        return entries
    indices = np.linspace(0, len(entries) - 1, limit, dtype=int)
    return [entries[i] for i in indices]


def image_tensor(image, device):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0).to(device)


def estimate_pair_transform(feat_prev, feat_curr, matcher, args):
    """Estimasi current -> previous; kembalikan metrik geometrik pair."""
    started = time.perf_counter()
    with torch.inference_mode():
        result = matcher({"image0": feat_prev, "image1": feat_curr})

    matches = result["matches"][0].detach().cpu().numpy()
    kp_prev = feat_prev["keypoints"][0].detach().cpu().numpy()
    kp_curr = feat_curr["keypoints"][0].detach().cpu().numpy()
    base = {
        "keypoints_prev": int(len(kp_prev)),
        "keypoints_curr": int(len(kp_curr)),
        "total_matches": int(len(matches)),
        "total_inliers": 0,
        "inlier_ratio": 0.0,
        "mean_inlier_reprojection_error_px": None,
        "median_inlier_reprojection_error_px": None,
        "transform_type": None,
        "transform_scale": None,
        "transform_angle_deg": None,
        "match_seconds": round(time.perf_counter() - started, 6),
        "accepted": False,
        "reject_reason": None,
    }
    if len(matches) < args.min_inliers:
        base["reject_reason"] = "matches_below_minimum"
        return None, base

    src = kp_curr[matches[:, 1]].astype(np.float32)
    dst = kp_prev[matches[:, 0]].astype(np.float32)
    matrix = None
    inlier_mask = None
    transform_type = None

    affine, affine_inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC,
        ransacReprojThreshold=args.ransac_thresh,
        maxIters=5000, confidence=0.995, refineIters=10,
    )
    if affine is not None and affine_inliers is not None:
        count = int(affine_inliers.sum())
        ratio = count / max(1, len(matches))
        if count >= args.min_inliers and ratio >= args.min_inlier_ratio:
            matrix = np.vstack([affine, [0.0, 0.0, 1.0]]).astype(np.float64)
            inlier_mask = affine_inliers.ravel().astype(bool)
            transform_type = "affine_partial"

    if matrix is None:
        homography, homography_inliers = cv2.findHomography(
            src, dst, cv2.RANSAC, args.ransac_thresh,
            maxIters=5000, confidence=0.995,
        )
        if homography is not None and homography_inliers is not None:
            count = int(homography_inliers.sum())
            ratio = count / max(1, len(matches))
            if count >= args.min_inliers and ratio >= args.min_inlier_ratio:
                matrix = homography.astype(np.float64)
                inlier_mask = homography_inliers.ravel().astype(bool)
                transform_type = "homography"

    if inlier_mask is not None:
        base["total_inliers"] = int(inlier_mask.sum())
        base["inlier_ratio"] = float(inlier_mask.sum() / max(1, len(matches)))

    if matrix is None or not np.isfinite(matrix).all():
        base["reject_reason"] = "ransac_or_inlier_threshold_failed"
        return None, base

    # Residual dihitung pada titik yang dipakai RANSAC: ini indikator fit internal,
    # bukan ground-truth geometric accuracy.
    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    residuals = np.linalg.norm(projected[inlier_mask] - dst[inlier_mask], axis=1)
    if residuals.size:
        base["mean_inlier_reprojection_error_px"] = float(np.mean(residuals))
        base["median_inlier_reprojection_error_px"] = float(np.median(residuals))

    a = matrix[:2, :2]
    scale = float(np.sqrt(abs(np.linalg.det(a))))
    angle = abs(float(np.degrees(np.arctan2(a[1, 0], a[0, 0]))))
    base["transform_type"] = transform_type
    base["transform_scale"] = scale
    base["transform_angle_deg"] = angle

    if not (0.65 <= scale <= 1.55):
        base["reject_reason"] = "scale_out_of_range"
        return None, base
    if angle > 18:
        base["reject_reason"] = "rotation_out_of_range"
        return None, base
    if abs(matrix[2, 0]) > 0.002 or abs(matrix[2, 1]) > 0.002:
        base["reject_reason"] = "perspective_out_of_range"
        return None, base

    base["accepted"] = True
    return matrix, base


def corners_of(image):
    h, w = image.shape[:2]
    return np.float32([
        [0, 0], [w, 0], [w, h], [0, h]
    ]).reshape(-1, 1, 2)


def build_transforms(images, extractor, matcher, device, args, debug_dir, track_id):
    """Bangun transform berantai dan kumpulkan metrik setiap pasangan crop."""
    pair_metrics = []
    if len(images) < 2:
        return ([(0, np.eye(3, dtype=np.float64))] if images else []), pair_metrics

    features = []
    for image in images:
        tensor = image_tensor(image, device)
        with torch.inference_mode():
            features.append(extractor.extract(tensor))
        del tensor

    center = len(images) // 2
    transforms = {center: np.eye(3, dtype=np.float64)}
    accepted = {center}

    for direction in (1, -1):
        current = center
        indices = range(center + 1, len(images)) if direction == 1 else range(center - 1, -1, -1)
        for index in indices:
            matrix, metrics = estimate_pair_transform(features[current], features[index], matcher, args)
            metrics.update({"track_id": int(track_id), "reference_crop": int(current), "candidate_crop": int(index)})

            if matrix is None:
                pair_metrics.append(metrics)
                print(
                    f"      [REJECT] track {track_id}, crop {index}: "
                    f"matches={metrics['total_matches']}, inliers={metrics['total_inliers']}, "
                    f"reason={metrics['reject_reason']}"
                )
                continue

            projected = cv2.perspectiveTransform(corners_of(images[index]), matrix)
            ph = projected[:, 0, 1].max() - projected[:, 0, 1].min()
            pw = projected[:, 0, 0].max() - projected[:, 0, 0].min()
            h, w = images[index].shape[:2]
            if not np.isfinite(projected).all() or pw > w * 2.5 or ph > h * 2.5:
                metrics["accepted"] = False
                metrics["reject_reason"] = "projected_bounds_extreme"
                pair_metrics.append(metrics)
                print(f"      [REJECT] track {track_id}, crop {index}: proyeksi ekstrem")
                continue

            transforms[index] = transforms[current] @ matrix
            accepted.add(index)
            pair_metrics.append(metrics)
            current = index
            print(
                f"      [MATCH] track {track_id}, crop {index}: "
                f"{metrics['total_matches']} matches, {metrics['total_inliers']} inliers, "
                f"ratio={metrics['inlier_ratio']:.3f}"
            )

            if debug_dir is not None:
                prev_index = metrics["reference_crop"]
                vis = np.hstack([
                    cv2.resize(images[prev_index], (320, 180)),
                    cv2.resize(images[index], (320, 180))
                ])
                cv2.imwrite(
                    str(debug_dir / f"track_{track_id:04d}_crop_{index:04d}.jpg"),
                    vis, [cv2.IMWRITE_JPEG_QUALITY, 80]
                )

    del features
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return [(i, transforms[i]) for i in sorted(accepted)], pair_metrics


def render_panorama(images, registered, max_mp, source_frames=None):
    """
    Render one panorama and estimate ONE container bounding box in panorama space.

    Each original YOLO detection bbox is converted from original-frame pixels
    into its resized crop coordinates, projected through the accumulated
    stitching transform, and then robustly aggregated into one container bbox.
    """
    if not registered:
        return None, None, 0

    all_corners = []
    for index, transform in registered:
        projected = cv2.perspectiveTransform(corners_of(images[index]), transform)
        if not np.isfinite(projected).all():
            return None, None, 0
        all_corners.append(projected.reshape(-1, 2))

    points = np.concatenate(all_corners, axis=0)
    min_x = int(np.floor(points[:, 0].min()))
    min_y = int(np.floor(points[:, 1].min()))
    max_x = int(np.ceil(points[:, 0].max()))
    max_y = int(np.ceil(points[:, 1].max()))
    width, height = max_x - min_x, max_y - min_y
    if width <= 0 or height <= 0:
        return None, None, 0

    scale = min(1.0, np.sqrt((max_mp * 1_000_000) / float(width * height)))
    out_w = max(1, int(round(width * scale)))
    out_h = max(1, int(round(height * scale)))
    shift = np.array([
        [scale, 0, -min_x * scale],
        [0, scale, -min_y * scale],
        [0, 0, 1],
    ], dtype=np.float64)

    accumulator = np.zeros((out_h, out_w, 3), dtype=np.float32)
    weight_sum = np.zeros((out_h, out_w), dtype=np.float32)

    for index, transform in registered:
        image = images[index]
        matrix = shift @ transform
        warped = cv2.warpPerspective(image, matrix, (out_w, out_h))
        mask_src = np.ones(image.shape[:2], dtype=np.uint8) * 255
        mask = cv2.warpPerspective(mask_src, matrix, (out_w, out_h))
        weight = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
        if weight.max() > 0:
            weight = weight / weight.max()
        weight = cv2.GaussianBlur(weight, (0, 0), 3)
        accumulator += warped.astype(np.float32) * weight[..., None]
        weight_sum += weight
        del warped, mask, mask_src, weight

    panorama = np.divide(
        accumulator, np.maximum(weight_sum[..., None], 1e-5)
    ).clip(0, 255).astype(np.uint8)

    occupied = np.where(weight_sum > 0.02)
    if len(occupied[0]) == 0:
        return None, None, 0
    crop_y1, crop_y2 = int(occupied[0].min()), int(occupied[0].max() + 1)
    crop_x1, crop_x2 = int(occupied[1].min()), int(occupied[1].max() + 1)
    panorama = panorama[crop_y1:crop_y2, crop_x1:crop_x2]

    crop_shift = np.array([
        [1, 0, -crop_x1],
        [0, 1, -crop_y1],
        [0, 0, 1],
    ], dtype=np.float64)
    final_panorama_size = (int(panorama.shape[1]), int(panorama.shape[0]))

    # One projected YOLO detection rectangle per registered source crop.
    projected_detection_boxes = []
    for index, transform in registered:
        if source_frames is None or index >= len(source_frames):
            continue
        source_info = source_frames[index]
        detection = source_info.get("video_detection_bbox")
        crop_bounds = source_info.get("video_crop_bbox")
        if not detection or not crop_bounds:
            continue

        crop_w_original = max(1.0, float(crop_bounds["width"]))
        crop_h_original = max(1.0, float(crop_bounds["height"]))
        crop_h_resized, crop_w_resized = images[index].shape[:2]

        # Convert original-video bbox coordinates into the resized crop coordinates.
        sx = crop_w_resized / crop_w_original
        sy = crop_h_resized / crop_h_original
        dx1 = (float(detection["x"]) - float(crop_bounds["x"])) * sx
        dy1 = (float(detection["y"]) - float(crop_bounds["y"])) * sy
        dx2 = (float(detection["x"]) + float(detection["width"]) -
               float(crop_bounds["x"])) * sx
        dy2 = (float(detection["y"]) + float(detection["height"]) -
               float(crop_bounds["y"])) * sy

        detection_corners = np.float32([
            [dx1, dy1], [dx2, dy1], [dx2, dy2], [dx1, dy2]
        ]).reshape(-1, 1, 2)

        final_transform = crop_shift @ shift @ transform
        projected = cv2.perspectiveTransform(detection_corners, final_transform)
        if not np.isfinite(projected).all():
            continue

        xy = projected.reshape(-1, 2)
        x1 = float(np.clip(xy[:, 0].min(), 0, final_panorama_size[0]))
        y1 = float(np.clip(xy[:, 1].min(), 0, final_panorama_size[1]))
        x2 = float(np.clip(xy[:, 0].max(), 0, final_panorama_size[0]))
        y2 = float(np.clip(xy[:, 1].max(), 0, final_panorama_size[1]))
        if x2 > x1 and y2 > y1:
            projected_detection_boxes.append([x1, y1, x2, y2])

    container_bbox = None
    if projected_detection_boxes:
        boxes = np.asarray(projected_detection_boxes, dtype=np.float64)

        # Union all projected per-frame container detections into ONE bbox.
        # x/y = top-left of the union; right/bottom = farthest extent.
        left = float(np.min(boxes[:, 0]))
        top = float(np.min(boxes[:, 1]))
        right = float(np.max(boxes[:, 2]))
        bottom = float(np.max(boxes[:, 3]))

        x1 = max(0, int(np.floor(left)))
        y1 = max(0, int(np.floor(top)))
        x2 = min(final_panorama_size[0], int(np.ceil(right)))
        y2 = min(final_panorama_size[1], int(np.ceil(bottom)))

        if x2 > x1 and y2 > y1:
            container_bbox = {
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1,
            }

    return panorama, container_bbox, len(projected_detection_boxes)


def main():
    args = apply_profile(parse_args())
    random.seed(args.seed)
    np.random.seed(args.seed)
    cv2.setNumThreads(2)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    source = Path(args.source).resolve()
    if not source.is_file():
        sys.exit(f"[ERROR] Video tidak ditemukan: {source}")

    model_path = Path(args.model).resolve()
    if not model_path.is_file():
        sys.exit(f"[ERROR] Model tidak ditemukan: {model_path}")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = out_dir / "_crops"
    debug_dir = out_dir / "_debug_matches" if args.debug_matches else None
    if args.save_crops:
        crop_dir.mkdir(parents=True, exist_ok=True)
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    classes = None
    if args.classes:
        classes = [int(x.strip()) for x in args.classes.split(",") if x.strip()]

    print("1. YOLO Tracking...")
    print(f"   video={source.name} | profile={args.profile}")
    print(f"   imgsz={args.imgsz} stride={args.vid_stride} conf={args.conf}")
    print(f"   crop_pad={args.crop_pad} max_crops/track={args.max_crops}")
    print(f"   max_crop_dim={args.max_crop_dim} device={args.device}")

    model = YOLO(str(model_path))
    results = model.track(
        source=str(source),
        tracker=args.tracker,
        conf=args.conf,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
        classes=classes,
        stream=True,
        persist=True,
        verbose=False,
        device=args.device,
    )

    # Simpan (nomor_frame, crop) agar urutan temporal tetap terjaga.
    tracks = {}
    track_seen = {}
    track_stats = defaultdict(lambda: {"frames": [], "confidences": [], "box_areas": []})
    frame_rows = []
    processed = 0
    total_detections = 0
    total_tracked_detections = 0
    rejected_crops = 0
    tracking_started = time.perf_counter()

    for result in results:
        processed += 1
        boxes_obj = result.boxes
        raw_detection_count = int(len(boxes_obj)) if boxes_obj is not None else 0
        total_detections += raw_detection_count
        frame_confidences = []
        frame_tracked_count = 0
        if boxes_obj is None or boxes_obj.id is None:
            frame_rows.append({"processed_frame": processed, "raw_detections": raw_detection_count,
                               "tracked_detections": 0, "mean_confidence": None})
            continue

        frame = result.orig_img
        boxes = boxes_obj.xyxy.detach().cpu().numpy()
        track_ids = boxes_obj.id.int().detach().cpu().tolist()
        confidences = boxes_obj.conf.detach().cpu().numpy() if boxes_obj.conf is not None else np.full(len(boxes), np.nan)
        frame_tracked_count = len(track_ids)
        total_tracked_detections += frame_tracked_count

        for box, track_id, confidence in zip(boxes, track_ids, confidences):
            track_stats[track_id]["frames"].append(processed)
            if np.isfinite(confidence):
                track_stats[track_id]["confidences"].append(float(confidence))
            x1, y1, x2, y2 = box
            track_stats[track_id]["box_areas"].append(float(max(0, x2-x1) * max(0, y2-y1)))
            frame_confidences.append(float(confidence) if np.isfinite(confidence) else None)
            crop, crop_bounds = make_crop(
                frame, box, args.crop_pad, args.max_crop_dim, return_bounds=True
            )
            if not valid_crop(crop, args.min_crop_size, args.min_crop_blur):
                rejected_crops += 1
                continue

            # processed is the 1-based index in the stride-sampled inference stream.
            # Map it back to the decoded source video's frame index.
            source_video_frame_0based = max(0, (processed - 1) * args.vid_stride)
            source_video_frame_1based = source_video_frame_0based + 1
            x1, y1, x2, y2 = [float(v) for v in box]
            frame_coordinate_info = {
                "processed_frame_index": int(processed),
                "source_video_frame_0based": int(source_video_frame_0based),
                "source_video_frame_1based": int(source_video_frame_1based),
                "video_detection_bbox": {
                    "x": round(x1, 2),
                    "y": round(y1, 2),
                    "width": round(max(0.0, x2 - x1), 2),
                    "height": round(max(0.0, y2 - y1), 2),
                },
                "video_crop_bbox": crop_bounds,
                "video_frame_size": {
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                },
            }

            entries = tracks.setdefault(track_id, [])
            seen = track_seen.get(track_id, 0) + 1
            track_seen[track_id] = seen

            # Reservoir sampling: each crop keeps its own video-frame coordinates.
            if len(entries) < args.max_crops:
                entries.append((processed, crop, frame_coordinate_info))
            else:
                replacement = random.randrange(seen)
                if replacement < args.max_crops:
                    entries[replacement] = (processed, crop, frame_coordinate_info)
            del crop

        valid_frame_conf = [v for v in frame_confidences if v is not None]
        frame_rows.append({
            "processed_frame": processed,
            "raw_detections": raw_detection_count,
            "tracked_detections": frame_tracked_count,
            "mean_confidence": safe_mean(valid_frame_conf),
        })

        if processed % 50 == 0:
            print(f"   frame {processed}: {len(tracks)} track")

    tracking_seconds = time.perf_counter() - tracking_started
    del results, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Urutkan berdasarkan frame dan ambil sampel merata.
    for track_id in list(tracks):
        tracks[track_id].sort(key=lambda x: x[0])
        tracks[track_id] = choose_ordered_samples(
            tracks[track_id], args.max_crops
        )

    track_rows = []
    for track_id, stats in track_stats.items():
        frame_ids = sorted(set(stats["frames"]))
        span = (frame_ids[-1] - frame_ids[0] + 1) if frame_ids else 0
        gaps = sum(1 for a, b in zip(frame_ids, frame_ids[1:]) if b - a > 1)
        track_rows.append({
            "track_id": int(track_id),
            "detections": len(frame_ids),
            "first_frame": frame_ids[0] if frame_ids else None,
            "last_frame": frame_ids[-1] if frame_ids else None,
            "temporal_span": span,
            "continuity_proxy": (len(frame_ids) / span) if span else None,
            "fragmentation_proxy": int(gaps),
            "mean_confidence": safe_mean(stats["confidences"]),
            "mean_box_area_px": safe_mean(stats["box_areas"]),
        })
    track_rows.sort(key=lambda row: row["track_id"])
    stage_rows = [{"stage": "yolo_tracking", "seconds": round(tracking_seconds, 4)}]
    match_rows = []
    panorama_rows = []
    error_rows = []

    print(
        f"2. Tracking selesai: {len(tracks)} track, "
        f"{processed} frame, {rejected_crops} crop ditolak"
    )
    print(
        f"   reservoir aktif: maksimal {args.max_crops} crop/track "
        "(batas memori diterapkan selama decoding)"
    )

    if args.save_crops:
        for track_id, entries in tracks.items():
            track_folder = crop_dir / f"track_{track_id:04d}"
            track_folder.mkdir(exist_ok=True)
            for frame_number, image, _frame_coordinates in entries:
                cv2.imwrite(
                    str(track_folder / f"frame_{frame_number:06d}.jpg"),
                    image, [cv2.IMWRITE_JPEG_QUALITY, 92]
                )

    device = torch.device(args.device)
    extractor = ALIKED(max_num_keypoints=args.keypoints).eval().to(device)
    matcher = LightGlue(features="aliked").eval().to(device)

    crop_count_skips = []
    valid_tracks = []
    for track_id, entries in tracks.items():
        if len(entries) < args.min_crops_per_track:
            crop_count_skips.append({
                "track_id": int(track_id),
                "crop_count": int(len(entries)),
                "required_minimum": int(args.min_crops_per_track),
                "reason": "insufficient_crop_count",
            })
            print(
                f"   [SKIP CROP COUNT] track {track_id}: "
                f"{len(entries)} crop < {args.min_crops_per_track}"
            )
            continue
        if len(entries) < args.min_track_len:
            continue
        valid_tracks.append((track_id, entries))

    print(
        f"3. Stitching ALIKED + LightGlue: {len(valid_tracks)} track "
        f"(minimum {args.min_crops_per_track} crop/track)"
    )
    saved = failed = 0

    for number, (track_id, entries) in enumerate(
        tqdm(valid_tracks, desc="Stitching", unit="track"), start=1
    ):
        output_path = out_dir / f"container_{track_id:04d}.jpg"
        selected_entries = choose_ordered_samples(entries, args.max_stitch_frames)
        source_frame_coordinates = [item[2] for item in selected_entries]
        images = [item[1] for item in selected_entries]

        print(
            f"\n[{number}/{len(valid_tracks)}] Track {track_id}: "
            f"{len(images)} crop"
        )

        track_stitch_started = time.perf_counter()
        try:
            registered, pair_metrics = build_transforms(
                images, extractor, matcher, device, args,
                debug_dir, track_id
            )
            match_rows.extend(pair_metrics)
            if len(registered) < 3:
                print("   [SKIP] Kurang dari 3 crop berhasil diregistrasi")
                error_rows.append({"stage": "stitching", "track_id": int(track_id), "pair_index": None,
                                   "error_type": "insufficient_registered_crops", "message": f"registered={len(registered)}"})
                failed += 1
                continue

            panorama, container_bbox, bbox_samples = render_panorama(
                images, registered, args.max_pano_mp,
                source_frames=source_frame_coordinates,
            )
            if panorama is None or panorama.size == 0:
                print("   [FAIL] Panorama kosong")
                error_rows.append({"stage": "rendering", "track_id": int(track_id), "pair_index": None,
                                   "error_type": "empty_panorama", "message": "render_panorama returned empty result"})
                failed += 1
                continue

            if container_bbox is None:
                print("   [SKIP] Bounding box kontainer tidak dapat diproyeksikan ke panorama")
                error_rows.append({
                    "stage": "container_bbox",
                    "track_id": int(track_id),
                    "pair_index": None,
                    "error_type": "container_bbox_projection_failed",
                    "message": "No valid projected YOLO detection boxes in panorama coordinates.",
                })
                failed += 1
                continue

            ok = cv2.imwrite(
                str(output_path), panorama,
                [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
            )
            if not ok:
                print(f"   [FAIL] Gagal menulis {output_path.name}")
                error_rows.append({"stage": "saving", "track_id": int(track_id), "pair_index": None,
                                   "error_type": "imwrite_failed", "message": str(output_path)})
                failed += 1
                continue

            file_size_bytes = output_path.stat().st_size if output_path.exists() else None
            panorama_rows.append({
                "track_id": int(track_id),
                "first_frame": int(entries[0][0]),
                "last_frame": int(entries[-1][0]),
                "input_crops": len(images),
                "registered_crops": len(registered),
                "container_bbox": container_bbox,
                "bbox_samples": int(bbox_samples),
                "registration_coverage": len(registered) / max(1, len(images)),
                "width_px": int(panorama.shape[1]),
                "height_px": int(panorama.shape[0]),
                "megapixels": round((panorama.shape[1] * panorama.shape[0]) / 1_000_000, 4),
                "file_size_bytes": file_size_bytes,
                "output_file": output_path.name,
            })

            print(
                f"   [SAVED] {output_path.name} | "
                f"{panorama.shape[1]}x{panorama.shape[0]} | "
                f"{len(registered)}/{len(images)} crop"
            )
            saved += 1
            del panorama

        except torch.cuda.OutOfMemoryError as error:
            print("   [CUDA OOM] Track dilewati")
            error_rows.append({"stage": "stitching", "track_id": int(track_id), "pair_index": None,
                               "error_type": "CUDAOutOfMemoryError", "message": str(error)})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            failed += 1
        except Exception as error:
            print(f"   [ERROR] Track {track_id}: {error}")
            error_rows.append({"stage": "stitching", "track_id": int(track_id), "pair_index": None,
                               "error_type": type(error).__name__, "message": str(error)})
            failed += 1
        finally:
            stage_rows.append({"stage": f"stitch_track_{track_id}", "seconds": round(time.perf_counter() - track_stitch_started, 4)})
            del images
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\nSelesai.")
    print(f"   panorama tersimpan : {saved}")
    print(f"   track gagal/dilewati: {failed}")
    print(f"   output              : {out_dir}")

    total_pairs = len(match_rows)
    accepted_pairs = sum(bool(row.get("accepted")) for row in match_rows)
    inlier_ratios = [row.get("inlier_ratio") for row in match_rows if row.get("inlier_ratio") is not None]
    reproj_errors = [row.get("mean_inlier_reprojection_error_px") for row in match_rows
                     if row.get("mean_inlier_reprojection_error_px") is not None]
    all_track_frames = sum(row["detections"] for row in track_rows)
    summary = {
        "input_video": str(source),
        "model": str(model_path),
        "profile": args.profile,
        "minimum_crops_per_track": int(args.min_crops_per_track),
        "parameters": {k: v for k, v in vars(args).items() if k not in {"source", "model", "output_dir"}},
        "tracking": {
            "processed_frames": processed,
            "raw_detections": total_detections,
            "tracked_detections": total_tracked_detections,
            "unique_track_ids": len(track_rows),
            "rejected_crops": rejected_crops,
            "tracks_skipped_low_crop_count": len(crop_count_skips),
            "mean_track_confidence": safe_mean([r["mean_confidence"] for r in track_rows]),
            "mean_track_continuity_proxy": safe_mean([r["continuity_proxy"] for r in track_rows]),
            "total_fragmentation_proxy": int(sum(r["fragmentation_proxy"] for r in track_rows)),
            "runtime_seconds": round(tracking_seconds, 4),
            "ground_truth_available": False,
            "accuracy_metrics": None,
            "accuracy_note": "MOTA, IDF1, HOTA, precision, dan recall membutuhkan anotasi ground-truth objek/track. Continuity dan fragmentation di sini hanya proxy internal.",
        },
        "aliked_lightglue": {
            "pair_attempts": total_pairs,
            "accepted_pairs": accepted_pairs,
            "rejected_pairs": total_pairs - accepted_pairs,
            "registration_success_rate": (accepted_pairs / total_pairs) if total_pairs else None,
            "mean_matches": safe_mean([r["total_matches"] for r in match_rows]),
            "mean_inliers": safe_mean([r["total_inliers"] for r in match_rows]),
            "mean_inlier_ratio": safe_mean(inlier_ratios),
            "mean_inlier_reprojection_error_px": safe_mean(reproj_errors),
            "mean_pair_matching_seconds": safe_mean([r["match_seconds"] for r in match_rows]),
            "geometric_accuracy_note": "Reprojection residual dihitung pada inlier RANSAC yang sama; ini mengukur fit internal, bukan akurasi absolut. Evaluasi akurasi geometrik memerlukan titik korespondensi/transformasi ground-truth independen.",
        },
        "stitching": {
            "panoramas_saved": saved,
            "tracks_failed_or_skipped": failed,
            "mean_registration_coverage": safe_mean([r["registration_coverage"] for r in panorama_rows]),
            "mean_panorama_megapixels": safe_mean([r["megapixels"] for r in panorama_rows]),
        },
        "system": {
            "device": str(args.device),
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and str(args.device).startswith("cuda") else None,
            "peak_cuda_memory_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2) if torch.cuda.is_available() else None,
        },
        "errors_count": len(error_rows),
    }
    write_csv(out_dir / "crop_count_skips.csv", crop_count_skips,
              ["track_id", "crop_count", "required_minimum", "reason"])
    # Arrange each panorama on a virtual grid in source-video appearance order.
    # x/y are positions on the shared layout canvas; width/height are the
    # actual dimensions of each saved panorama (not fixed 4000x4000 cells).
    ordered_panorama_rows = sorted(
        panorama_rows,
        key=lambda row: (row.get("first_frame", 0), row.get("track_id", 0)),
    )

    container_coordinate_rows = []
    container_count = len(ordered_panorama_rows)

    if container_count:
        grid_columns = int(math.ceil(math.sqrt(container_count)))
        grid_rows = int(math.ceil(container_count / grid_columns))

        # Equal grid slots prevent overlap while allowing each panorama
        # to retain its native aspect ratio and pixel dimensions.
        cell_width = max(int(row["width_px"]) for row in ordered_panorama_rows)
        cell_height = max(int(row["height_px"]) for row in ordered_panorama_rows)

        for index, panorama_row in enumerate(ordered_panorama_rows):
            row_index = index // grid_columns
            column_index = index % grid_columns

            pano_width = int(panorama_row["width_px"])
            pano_height = int(panorama_row["height_px"])

            container_coordinate_rows.append({
                "container_id": int(panorama_row["track_id"]),
                "container_file": panorama_row["output_file"],
                "x": int(column_index * cell_width),
                "y": int(row_index * cell_height),
                "width": pano_width,
                "height": pano_height,
            })

        layout_canvas_width = int(grid_columns * cell_width)
        layout_canvas_height = int(grid_rows * cell_height)
    else:
        grid_columns = grid_rows = 0
        layout_canvas_width = layout_canvas_height = 0

    coordinate_document = {
        "canvas_width": layout_canvas_width,
        "canvas_height": layout_canvas_height,
        "containers": container_coordinate_rows,
    }
    with (out_dir / "container_coordinates.json").open("w", encoding="utf-8") as jf:
        json.dump(coordinate_document, jf, indent=2, ensure_ascii=False)

    write_csv(
        out_dir / "container_coordinates.csv",
        container_coordinate_rows,
        ["container_id", "container_file", "x", "y", "width", "height"],
    )
    export_analytics(out_dir, summary, frame_rows, track_rows, match_rows, panorama_rows, error_rows, stage_rows)


if __name__ == "__main__":
    main()
