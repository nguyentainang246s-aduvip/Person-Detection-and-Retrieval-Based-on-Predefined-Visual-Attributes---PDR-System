"""
Hệ thống tạo nhãn Ground-Truth bán tự động (Assisted Ground-Truth Generator)
Hỗ trợ tạo nhanh Ground-Truth cho video kiểm thử:
  1. Chạy phát hiện & tracking sơ bộ trên video
  2. Gom nhóm các tracklet và lưu ảnh đại diện (crop)
  3. Cho phép gán nhãn Ground-Truth (True Person ID, Gender, Colors, Hat, Glasses, Bag)
  4. Tự động nội suy và xuất file Ground-Truth chuẩn CSV (MOT Challenge format)
"""

import os
import sys
import argparse
import csv
import json
import cv2
import numpy as np

# Thêm thư mục gốc vào PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.detection.detector import PersonDetector
from src.tracking.tracker import PersonTracker
from src.utils.video_utils import open_video, read_frame


def extract_tracks_and_samples(video_path: str, output_dir: str, conf_thresh: float = 0.45):
    """Trích xuất tất cả các tracklet và lưu 3 ảnh đại diện cho mỗi tracklet."""
    os.makedirs(output_dir, exist_ok=True)
    crops_dir = os.path.join(output_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    tracker = PersonTracker(confidence_threshold=conf_thresh, device="cpu")
    cap, info = open_video(video_path)
    total_frames = info["total_frames"]
    fps = info["fps"] or 30.0

    print(f"[*] Đang phân tích video: {video_path}")
    print(f"[*] Độ phân giải: {info['width']}x{info['height']}, {fps:.1f} FPS, {total_frames} frames")

    raw_detections = [] # (frame_idx, track_id, x, y, w, h, conf)
    track_crops = {}     # track_id -> [crops]
    track_stats = {}     # track_id -> {'first': int, 'last': int, 'count': int}

    frame_idx = 0
    while True:
        ret, frame = read_frame(cap)
        if not ret:
            break

        objects = tracker.track(frame, persist=True)
        for obj in objects:
            tid = obj["track_id"]
            box = obj["bbox"] # [x1, y1, x2, y2]
            x1, y1, x2, y2 = [int(v) for v in box]
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            conf = float(obj.get("confidence", 0.5))

            raw_detections.append((frame_idx, tid, x1, y1, w, h, conf))

            if tid not in track_stats:
                track_stats[tid] = {"first": frame_idx, "last": frame_idx, "count": 1}
                track_crops[tid] = []
            else:
                track_stats[tid]["last"] = frame_idx
                track_stats[tid]["count"] += 1

            # Lưu ảnh crop (tối đa 3 ảnh: đầu, giữa, cuối)
            if len(track_crops[tid]) < 3 and h > 40 and w > 20:
                crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                if crop.size > 0:
                    track_crops[tid].append(crop)

        frame_idx += 1
        if frame_idx % 50 == 0 or frame_idx == total_frames:
            print(f"  -> Tiến độ: {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%)")

    cap.release()

    # Lưu ảnh crop mẫu ra đĩa
    for tid, crops in track_crops.items():
        for i, crop in enumerate(crops):
            crop_path = os.path.join(crops_dir, f"track_{tid:03d}_sample_{i+1}.jpg")
            cv2.imwrite(crop_path, crop)

    # Lưu thông tin track sơ bộ dạng JSON để người dùng dễ xem
    summary_path = os.path.join(output_dir, "tracks_summary.json")
    summary = {
        "video_path": video_path,
        "total_frames": total_frames,
        "total_raw_tracks": len(track_stats),
        "tracks": [
            {
                "track_id": tid,
                "first_frame": stats["first"],
                "last_frame": stats["last"],
                "duration_frames": stats["last"] - stats["first"] + 1,
                "detection_count": stats["count"]
            }
            for tid, stats in sorted(track_stats.items(), key=lambda x: x[1]["first"])
        ]
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Lưu raw detections
    raw_csv = os.path.join(output_dir, "raw_detections.csv")
    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "track_id", "x", "y", "w", "h", "conf"])
        writer.writerows(raw_detections)

    print(f"\n[+] Hoàn tất trích xuất!")
    print(f"[+] Tìm thấy {len(track_stats)} tracklets sơ bộ.")
    print(f"[+] Dữ liệu đã lưu tại: {output_dir}")
    print(f"    - Bảng tổng hợp: {summary_path}")
    print(f"    - Tọa độ bbox:   {raw_csv}")
    print(f"    - Ảnh mẫu người: {crops_dir}")
    return summary, raw_detections


def build_ground_truth_csv(raw_csv_path: str, id_mapping: dict, person_attributes: dict, output_gt_path: str):
    """
    Biên tập file Ground-Truth hoàn chỉnh từ raw detections + bản đồ mapping.

    Args:
        raw_csv_path: file raw_detections.csv
        id_mapping: ánh xạ {track_id_sơ_bộ: true_person_id} (để gộp các ID switch về cùng 1 người)
        person_attributes: {true_person_id: {'gender':..., 'upper_color':..., ...}}
        output_gt_path: đường dẫn lưu file gt.csv
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_gt_path)), exist_ok=True)

    rows_written = 0
    with open(raw_csv_path, "r", encoding="utf-8") as infile, \
         open(output_gt_path, "w", newline="", encoding="utf-8") as outfile:

        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)
        writer.writerow([
            "frame_idx", "person_id", "x", "y", "w", "h",
            "gender", "upper_color", "lower_color", "hat", "glasses", "backpack"
        ])

        for row in reader:
            raw_tid = int(row["track_id"])
            # Nếu track này được mapping tới một người thật (bỏ qua false positives nếu không map)
            if raw_tid in id_mapping:
                true_pid = id_mapping[raw_tid]
                attrs = person_attributes.get(true_pid, {})
                writer.writerow([
                    row["frame_idx"],
                    true_pid,
                    row["x"], row["y"], row["w"], row["h"],
                    attrs.get("gender", "Male"),
                    attrs.get("upper_color", "Other"),
                    attrs.get("lower_color", "Other"),
                    1 if attrs.get("hat", False) else 0,
                    1 if attrs.get("glasses", False) else 0,
                    1 if attrs.get("backpack", False) else 0
                ])
                rows_written += 1

    print(f"[+] Đã tạo file Ground-Truth: {output_gt_path} ({rows_written} dòng nhãn)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Công cụ hỗ trợ tạo Ground-Truth cho video PDR-System")
    parser.add_argument("--video", required=True, help="Đường dẫn file video test")
    parser.add_argument("--output_dir", default="data/gt/extract", help="Thư mục xuất dữ liệu trích xuất")
    args = parser.parse_args()

    extract_tracks_and_samples(args.video, args.output_dir)
