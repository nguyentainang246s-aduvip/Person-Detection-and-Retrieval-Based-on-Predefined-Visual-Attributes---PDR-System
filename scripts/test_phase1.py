"""
scripts/test_phase1.py
=======================
PHASE 1 – CHECKPOINT SCRIPT

Script kiểm tra toàn bộ môi trường và kiến thức Phase 1.
Chạy script này để xác nhận Phase 1 đã hoàn thành.

CÁCH CHẠY:
    (Từ thư mục gốc project)
    .\\venv\\Scripts\\python.exe scripts/test_phase1.py

EXPECTED OUTPUT khi thành công:
    [✓] Python version: 3.12.x
    [✓] OpenCV version: 4.x.x
    [✓] NumPy version: 1.x.x
    [✓] PyTorch version: 2.x.x (CUDA available: True)
    [✓] Ultralytics version: 8.x.x
    [✓] Video utils module: OK
    [✓] Video test: Opened OK - 1920x1080 @ 30fps, 300 frames
    [✓] Frame reading: OK - shape (1080, 1920, 3)
    [✓] FPS counter: OK
    [✓] Person crop: OK - crop shape (200, 150, 3)

    ============================================
    ✅ PHASE 1 CHECKPOINT: PASSED
    ============================================
"""

import sys
import os

# Thêm thư mục gốc project vào Python path để import được src/*
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def check(condition, msg_ok, msg_fail=""):
    """Helper in kết quả check."""
    if condition:
        print(f"  [✓] {msg_ok}")
        return True
    else:
        print(f"  [✗] FAILED: {msg_fail}")
        return False


def test_python():
    print("\n-- Kiem tra Python --")
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 10
    check(ok, f"Python version: {v.major}.{v.minor}.{v.micro}", "Cần Python 3.10+")
    return ok


def test_opencv():
    print("\n-- Kiem tra OpenCV --")
    try:
        import cv2
        check(True, f"OpenCV version: {cv2.__version__}")
        return True
    except ImportError:
        check(False, "", "OpenCV chưa cài. Chạy: pip install opencv-python")
        return False


def test_numpy():
    print("\n-- Kiem tra NumPy --")
    try:
        import numpy as np
        check(True, f"NumPy version: {np.__version__}")
        return True
    except ImportError:
        check(False, "", "NumPy chưa cài.")
        return False


def test_torch():
    print("\n-- Kiem tra PyTorch + CUDA --")
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "N/A"
        check(True, f"PyTorch version: {torch.__version__}")
        if cuda_ok:
            check(True, f"CUDA available: True | GPU: {gpu_name}")
        else:
            print(f"  [!] CUDA không khả dụng. Chạy trên CPU (chậm hơn nhưng vẫn được)")
        return True
    except ImportError:
        check(False, "", "PyTorch chưa cài. Xem hướng dẫn cài bên dưới.")
        return False


def test_ultralytics():
    print("\n-- Kiem tra Ultralytics (YOLO) --")
    try:
        import ultralytics
        check(True, f"Ultralytics version: {ultralytics.__version__}")
        return True
    except ImportError:
        check(False, "", "Ultralytics chưa cài. Chạy: pip install ultralytics")
        return False


def test_yaml():
    print("\n-- Kiem tra PyYAML + Config --")
    try:
        import yaml
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config", "config.yaml"
        )
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        check(True, f"Config loaded: {list(config.keys())}")
        return True
    except FileNotFoundError:
        check(False, "", "Không tìm thấy config/config.yaml")
        return False
    except ImportError:
        check(False, "", "PyYAML chưa cài. Chạy: pip install pyyaml")
        return False


def test_video_utils():
    print("\n-- Kiem tra Video Utils Module --")
    try:
        from src.utils.video_utils import (
            open_video, read_frame, crop_person,
            frame_to_timestamp, resize_frame, bgr_to_rgb, FPSCounter
        )
        check(True, "Video utils module import: OK")
        return True
    except ImportError as e:
        check(False, "", f"Import lỗi: {e}")
        return False


def test_video_processing():
    """
    Test doc video that hoac tao video gia de test.
    """
    print("\n-- Kiem tra Xu ly Video --")
    try:
        import cv2
        from src.utils.video_utils import (
            open_video, read_frame, crop_person,
            frame_to_timestamp, FPSCounter, bgr_to_rgb
        )

        # ── Test 1: Tạo video giả bằng numpy để test (không cần video thật)
        # Tạo frame giả: 480x640 pixel, màu ngẫu nhiên
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        check(True, f"Tạo frame giả thành công: shape={dummy_frame.shape}")

        # ── Test 2: BGR to RGB conversion
        rgb_frame = bgr_to_rgb(dummy_frame)
        ok = rgb_frame.shape == dummy_frame.shape
        check(ok, f"BGR→RGB conversion: shape {rgb_frame.shape}")

        # ── Test 3: Crop person
        # Giả lập bounding box: người ở vị trí x1=50, y1=30, x2=200, y2=400
        fake_bbox = [50, 30, 200, 400]
        crop = crop_person(dummy_frame, fake_bbox)
        expected_h = 400 - 30  # = 370
        expected_w = 200 - 50  # = 150
        ok = crop is not None and crop.shape == (expected_h, expected_w, 3)
        check(ok, f"Person crop: shape={crop.shape if crop is not None else None}")

        # ── Test 4: Timestamp conversion
        ts = frame_to_timestamp(2760, 30.0)
        ok = ts == "00:01:32"
        check(ok, f"Frame→Timestamp: frame 2760 @ 30fps = '{ts}'")

        # ── Test 5: FPS Counter
        import time
        fps_counter = FPSCounter(avg_window=5)
        for _ in range(5):
            fps_counter.start_frame()
            time.sleep(0.01)  # Giả lập xử lý 10ms/frame
            fps_counter.end_frame()
        fps = fps_counter.get_fps()
        ok = fps > 0
        check(ok, f"FPS Counter: {fps:.1f} FPS")

        # ── Test 6: Thử mở video thật nếu có
        video_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "test_videos"
        )
        video_files = [f for f in os.listdir(video_dir)
                       if f.endswith(('.mp4', '.avi', '.mkv', '.mov'))]

        if video_files:
            video_path = os.path.join(video_dir, video_files[0])
            cap, info = open_video(video_path)
            success, frame = read_frame(cap)
            cap.release()
            check(success, f"Video thật: {video_files[0]} | {info['width']}x{info['height']} @ {info['fps']:.0f}fps")
        else:
            print("  [!] Không có video test. Đặt video vào data/test_videos/ để test thêm.")

        return True

    except Exception as e:
        check(False, "", f"Lỗi: {e}")
        return False


def main():
    print("=" * 50)
    print("  PHASE 1 CHECKPOINT - ENVIRONMENT TEST")
    print("=" * 50)

    results = []
    results.append(test_python())
    results.append(test_opencv())
    results.append(test_numpy())
    results.append(test_torch())
    results.append(test_ultralytics())
    results.append(test_yaml())
    results.append(test_video_utils())
    results.append(test_video_processing())

    # ── Tổng kết ──
    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"  [PASSED] PHASE 1 CHECKPOINT: {passed}/{total}")
        print("  -> San sang chuyen sang PHASE 2: YOLO Detection!")
    else:
        print(f"  [PARTIAL] PHASE 1 CHECKPOINT: {passed}/{total} passed")
        print("  -> Xem loi ben tren va sua truoc khi tiep tuc.")
    print("=" * 50)


if __name__ == "__main__":
    main()
