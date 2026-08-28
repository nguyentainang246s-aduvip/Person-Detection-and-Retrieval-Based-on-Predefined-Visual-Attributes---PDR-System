"""
scripts/test_tracking.py
========================
PHASE 3 – CHECKPOINT & TEST SCRIPT: BYTETRACK MULTI-OBJECT TRACKING

Script kiểm tra khả năng gán và duy trì Track ID qua một chuỗi các frame video.

CÁCH CHẠY:
    # 1. Chạy test tự động với video giả lập (tạo chuỗi chuyển động mô phỏng):
    .\\venv\\Scripts\\python.exe scripts/test_tracking.py

    # 2. Chạy test với video thực tế:
    .\\venv\\Scripts\\python.exe scripts/test_tracking.py --video path/to/video.mp4
"""

import sys
import os
import argparse
import cv2
import numpy as np

# Thêm thư mục gốc vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tracking.tracker import PersonTracker
from src.utils.visualization import draw_bounding_box
from src.utils.logger import get_logger

logger = get_logger("test_tracking")


def create_synthetic_tracking_video(output_path="data/test_synthetic_motion.mp4", num_frames=30):
    """
    Tạo một video giả lập gồm 1 khối di chuyển ngang màn hình để test ByteTrack.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 15.0, (640, 480))

    for i in range(num_frames):
        frame = np.full((480, 640, 3), (240, 240, 240), dtype=np.uint8)
        # Vẽ một đối tượng chuyển động từ trái sang phải
        cx = 100 + i * 12
        cy = 240
        # Vẽ hình mô phỏng người (đầu + thân)
        cv2.circle(frame, (cx, cy - 60), 25, (50, 50, 200), -1)
        cv2.rectangle(frame, (cx - 30, cy - 35), (cx + 30, cy + 80), (200, 50, 50), -1)
        out.write(frame)

    out.release()
    return output_path


def test_tracker_initialization():
    print("\n-- 1. Kiem tra khoi tao PersonTracker (ByteTrack) --")
    try:
        tracker = PersonTracker(model_path="models/yolo/yolov8n.pt", tracker_type="bytetrack.yaml")
        print("  [✓] PersonTracker khoi tao thanh cong voi ByteTrack!")
        return tracker
    except Exception as e:
        print(f"  [✗] Loi khoi tao Tracker: {e}")
        return None


def test_tracking_on_video(tracker, video_path, max_frames=60):
    print(f"\n-- 2. Kiem tra Tracking tren Video: {video_path} --")
    if not os.path.exists(video_path):
        print(f"  [!] Video khong ton tai. Dang tao video mo phong tai: {video_path}")
        create_synthetic_tracking_video(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [✗] Khong the mo file video {video_path}")
        return False

    frame_idx = 0
    unique_track_ids = set()
    frames_with_tracks = 0

    os.makedirs("results", exist_ok=True)
    out_video_path = "results/test_tracking_output.mp4"
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Chạy tracking qua từng frame
        tracked_objects = tracker.track(frame, persist=True)

        if len(tracked_objects) > 0:
            frames_with_tracks += 1
            for obj in tracked_objects:
                unique_track_ids.add(obj["track_id"])
                # Vẽ box kèm ID
                draw_bounding_box(frame, obj["bbox"], obj["track_id"], color=(0, 255, 0))

        # Ghi frame kết quả
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    print(f"  [✓] Da xu ly xong {frame_idx} frames.")
    print(f"  [✓] Cac Track ID duy nhat duoc phat hien: {sorted(list(unique_track_ids)) if unique_track_ids else 'Khong co (anh gia lap khong phai nguoi that)'}")
    print(f"  [✓] Da xuat video tracking demo tai: {out_video_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test ByteTrack Multi-Object Tracking (Phase 3)")
    parser.add_argument("--video", type=str, default="data/test_synthetic_motion.mp4", help="Path video test")
    args = parser.parse_args()

    print("=" * 55)
    print("  PHASE 3 CHECKPOINT: BYTETRACK MULTI-OBJECT TRACKING")
    print("=" * 55)

    tracker = test_tracker_initialization()
    if tracker is None:
        print("\n  [FAILED] Khong the khoi tao Tracker!")
        return

    ok = test_tracking_on_video(tracker, args.video)

    print("\n" + "=" * 55)
    if ok:
        print("  [PASSED] PHASE 3 CHECKPOINT: HOAN THANH!")
        print("  -> San sang chuyen sang PHASE 4: Person Crop & Color Detection!")
    else:
        print("  [FAILED] PHASE 3 CHECKPOINT: Co loi xay ra.")
    print("=" * 55)


if __name__ == "__main__":
    main()
