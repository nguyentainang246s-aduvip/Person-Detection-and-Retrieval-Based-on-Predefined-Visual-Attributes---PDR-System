"""
scripts/test_color.py
=====================
PHASE 4 – CHECKPOINT & TEST SCRIPT: PERSON CROP & COLOR DETECTION

Kiểm tra khả năng cắt vùng người (Person ROI Crop) và nhận dạng chính xác
màu áo (Upper Color) & màu quần (Lower Color) bằng không gian màu HSV + K-Means.

CÁCH CHẠY:
    .\\venv\\Scripts\\python.exe scripts/test_color.py
"""

import sys
import os
import cv2
import numpy as np

# Thêm thư mục gốc vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attributes.color_detector import ColorDetector
from src.utils.video_utils import crop_person


def create_synthetic_person_image(upper_bgr, lower_bgr, width=120, height=300):
    """
    Tạo ảnh mô phỏng một người với màu áo và màu quần chỉ định.
    - 0% - 15%: Đầu/cổ
    - 15% - 55%: Thân trên (áo)
    - 55% - 90%: Thân dưới (quần)
    - 90% - 100%: Chân/giày
    """
    crop = np.full((height, width, 3), (200, 200, 200), dtype=np.uint8)

    # 1. Đầu (da)
    cv2.circle(crop, (width // 2, int(height * 0.08)), int(width * 0.18), (180, 200, 230), -1)

    # 2. Áo (Upper Body: 15% - 55%)
    y_up_start, y_up_end = int(height * 0.15), int(height * 0.55)
    crop[y_up_start:y_up_end, int(width * 0.1):int(width * 0.9)] = upper_bgr

    # 3. Quần (Lower Body: 55% - 90%)
    y_low_start, y_low_end = int(height * 0.55), int(height * 0.90)
    crop[y_low_start:y_low_end, int(width * 0.18):int(width * 0.82)] = lower_bgr

    return crop


def test_color_detection_cases():
    print("\n-- 1. Kiem tra ColorDetector tren cac mau sac chuan --")
    detector = ColorDetector(n_clusters=3)

    # Định nghĩa các test case (Màu áo BGR, Màu quần BGR, Expected Upper, Expected Lower)
    test_cases = [
        # BGR values: (Blue, Green, Red)
        ((0, 0, 240), (230, 0, 0), "Red", "Blue"),       # Áo Đỏ, Quần Xanh dương
        ((15, 15, 15), (15, 15, 15), "Black", "Black"),   # Áo Đen, Quần Đen
        ((245, 245, 245), (20, 20, 20), "White", "Black"), # Áo Trắng, Quần Đen
        ((0, 200, 0), (128, 128, 128), "Green", "Gray"), # Áo Xanh lá, Quần Xám
        ((0, 220, 220), (220, 10, 10), "Yellow", "Blue")  # Áo Vàng, Quần Xanh
    ]

    all_passed = True
    os.makedirs("results/crops", exist_ok=True)

    for i, (up_bgr, low_bgr, exp_up, exp_low) in enumerate(test_cases):
        person_crop = create_synthetic_person_image(up_bgr, low_bgr)
        # Lưu crop để kiểm chứng
        cv2.imwrite(f"results/crops/test_person_color_{i+1}.jpg", person_crop)

        result = detector.detect_colors(person_crop)
        pred_up = result["upper_color"]
        pred_low = result["lower_color"]

        up_match = (pred_up == exp_up)
        low_match = (pred_low == exp_low)

        status = "✓" if (up_match and low_match) else "✗"
        if not (up_match and low_match):
            all_passed = False

        print(f"  [{status}] Case #{i+1}: Expected=(Upper: {exp_up}, Lower: {exp_low}) -> Predicted=(Upper: {pred_up}, Lower: {pred_low})")

    return all_passed


def test_crop_pipeline_integration():
    print("\n-- 2. Kiem tra tich hop BBox Crop + Color Detection --")
    # Tạo 1 frame to 640x480 có 1 người bên trong tại bbox [100, 50, 220, 350]
    frame = np.full((480, 640, 3), (255, 255, 255), dtype=np.uint8)
    fake_person = create_synthetic_person_image((0, 0, 255), (255, 0, 0), width=120, height=300)
    frame[50:350, 100:220] = fake_person

    bbox = [100, 50, 220, 350]
    crop = crop_person(frame, bbox)

    if crop is None or crop.shape != (300, 120, 3):
        print(f"  [✗] Crop person that bai. Shape={getattr(crop, 'shape', None)}")
        return False

    detector = ColorDetector()
    colors = detector.detect_colors(crop)
    print(f"  [✓] Crop tu bbox thanh cong: shape={crop.shape} | Colors: {colors}")
    return True


def main():
    print("=" * 60)
    print("  PHASE 4 CHECKPOINT: PERSON CROP & COLOR DETECTION")
    print("=" * 60)

    ok1 = test_color_detection_cases()
    ok2 = test_crop_pipeline_integration()

    print("\n" + "=" * 60)
    if ok1 and ok2:
        print("  [PASSED] PHASE 4 CHECKPOINT: HOAN THANH!")
        print("  -> San sang chuyen sang PHASE 5: PAR Model Training & Fine-tuning!")
    else:
        print("  [FAILED] PHASE 4 CHECKPOINT: Mot so test case chua khop.")
    print("=" * 60)


if __name__ == "__main__":
    main()
