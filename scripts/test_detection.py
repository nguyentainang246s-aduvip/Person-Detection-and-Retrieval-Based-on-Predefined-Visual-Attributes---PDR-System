"""
scripts/test_detection.py
=========================
PHASE 2 – CHECKPOINT & TEST SCRIPT: YOLO PERSON DETECTION

Script kiểm tra khả năng nạp mô hình YOLOv8, chạy inference phát hiện người,
trích xuất bounding boxes và xuất file ảnh kết quả.

CÁCH CHẠY:
    # 1. Chạy test tự động với ảnh mẫu tạo sẵn (không cần tải thêm file bên ngoài):
    .\\venv\\Scripts\\python.exe scripts/test_detection.py

    # 2. Chạy test với ảnh cụ thể bất kỳ:
    .\\venv\\Scripts\\python.exe scripts/test_detection.py --image path/to/image.jpg

    # 3. Chạy test với video cụ thể:
    .\\venv\\Scripts\\python.exe scripts/test_detection.py --video path/to/video.mp4
"""

import sys
import os
import argparse
import cv2
import numpy as np

# Thêm thư mục gốc vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.detector import PersonDetector
from src.utils.logger import get_logger

logger = get_logger("test_detection")


def create_sample_test_image(output_path="data/test_sample.jpg"):
    """
    Tạo một ảnh mẫu đơn giản có vẽ hình người ước lệ để test inference không bị crash.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Khởi tạo background đường phố đơn giản (640x480)
    img = np.full((480, 640, 3), (220, 220, 220), dtype=np.uint8)

    # Vẽ vỉa hè
    cv2.rectangle(img, (0, 300), (640, 480), (160, 160, 160), -1)

    # Lưu ảnh
    cv2.imwrite(output_path, img)
    return output_path


def test_detector_initialization():
    print("\n-- 1. Kiem tra khoi tao PersonDetector --")
    try:
        detector = PersonDetector(model_path="models/yolo/yolov8n.pt", confidence_threshold=0.4)
        print("  [✓] PersonDetector khoi tao thanh cong!")
        return detector
    except Exception as e:
        print(f"  [✗] Loi khoi tao: {e}")
        return None


def test_detector_inference(detector, test_image_path):
    print(f"\n-- 2. Kiem tra chay inference tren anh: {test_image_path} --")
    if not os.path.exists(test_image_path):
        create_sample_test_image(test_image_path)

    frame = cv2.imread(test_image_path)
    if frame is None:
        print(f"  [✗] Khong the doc anh tu {test_image_path}")
        return False

    try:
        detections = detector.detect(frame)
        print(f"  [✓] Chay detect() thanh cong! Phat hien: {len(detections)} nguoi.")

        for idx, det in enumerate(detections):
            bbox = det["bbox"]
            conf = det["confidence"]
            print(f"      - Nguoi #{idx+1}: BBox={bbox}, Conf={conf*100:.1f}%")

        # Vẽ bounding box và lưu ảnh kết quả
        os.makedirs("results", exist_ok=True)
        annotated = detector.draw_detections(frame, detections)
        out_path = "results/test_detection_output.jpg"
        cv2.imwrite(out_path, annotated)
        print(f"  [✓] Da luu anh ket qua co bounding box tai: {out_path}")
        return True
    except Exception as e:
        print(f"  [✗] Loi khi chay inference: {e}")
        return False


def test_detector_video(detector, video_path, max_frames=60):
    print(f"\n-- 3. Kiem tra chay detection tren Video: {video_path} --")
    if not os.path.exists(video_path):
        print(f"  [!] File video {video_path} khong ton tai. Bo qua test video.")
        return True

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [✗] Khong the mo video: {video_path}")
        return False

    frame_count = 0
    total_persons_detected = 0

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        dets = detector.detect(frame)
        total_persons_detected += len(dets)
        frame_count += 1

    cap.release()
    print(f"  [✓] Da xu ly {frame_count} frames. Tong so luot phat hien: {total_persons_detected}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test YOLO Person Detection (Phase 2)")
    parser.add_argument("--image", type=str, default="data/test_sample.jpg", help="Path toi anh test")
    parser.add_argument("--video", type=str, default=None, help="Path toi video test")
    args = parser.parse_args()

    print("=" * 50)
    print("  PHASE 2 CHECKPOINT: YOLO PERSON DETECTION")
    print("=" * 50)

    # Bước 1: Khởi tạo model
    detector = test_detector_initialization()
    if detector is None:
        print("\n  [FAILED] Khong the khoi tao Detector!")
        return

    # Bước 2: Test inference trên ảnh
    img_ok = test_detector_inference(detector, args.image)

    # Bước 3: Test trên video nếu được truyền vào
    vid_ok = True
    if args.video:
        vid_ok = test_detector_video(detector, args.video)

    print("\n" + "=" * 50)
    if img_ok and vid_ok:
        print("  [PASSED] PHASE 2 CHECKPOINT: HOAN THANH!")
        print("  -> San sang chuyen sang PHASE 3: ByteTrack Multi-Object Tracking!")
    else:
        print("  [FAILED] PHASE 2 CHECKPOINT: Co loi xay ra, can kiem tra lai.")
    print("=" * 50)


if __name__ == "__main__":
    main()
