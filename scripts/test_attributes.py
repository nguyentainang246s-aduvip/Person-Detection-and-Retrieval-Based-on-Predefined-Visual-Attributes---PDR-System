"""
scripts/test_attributes.py
==========================
PHASE 5 – CHECKPOINT & TEST SCRIPT: PERSON ATTRIBUTE RECOGNITION (PAR)

Kiểm tra khả năng tích hợp mô hình ResNet50 dự đoán thuộc tính người:
Gender, Hat, Glasses, Backpack kết hợp với Color Detector (Áo, Quần).

CÁCH CHẠY:
    # 1. Chạy test tự động với ảnh mẫu:
    .\\venv\\Scripts\\python.exe scripts/test_attributes.py

    # 2. Chạy test với ảnh crop cụ thể:
    .\\venv\\Scripts\\python.exe scripts/test_attributes.py --image path/to/person_crop.jpg
"""

import sys
import os
import argparse
import cv2
import numpy as np

# Thêm thư mục gốc vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attributes.par_model import AttributeRecognizer
from src.attributes.color_detector import ColorDetector


def test_attribute_recognition(image_path=None):
    print("\n-- 1. Kiem tra khoi tao AttributeRecognizer & ColorDetector --")
    try:
        par_engine = AttributeRecognizer(weights_path="models/par/par_resnet50.pth")
        color_engine = ColorDetector()
        print("  [✓] Khoi tao thanh cong cac Engine phan tich thuoc tinh!")
    except Exception as e:
        print(f"  [✗] Loi khoi tao: {e}")
        return False

    print("\n-- 2. Kiem tra du doan thuoc tinh tren anh Person Crop --")
    if image_path and os.path.exists(image_path):
        crop = cv2.imread(image_path)
    else:
        # Lay anh mau da tao o Phase 4 hoac tao anh moi
        sample_path = "results/crops/test_person_color_1.jpg"
        if os.path.exists(sample_path):
            crop = cv2.imread(sample_path)
        else:
            crop = np.full((300, 150, 3), (180, 180, 180), dtype=np.uint8)

    if crop is None:
        print("  [✗] Khong the doc anh crop.")
        return False

    # 1. Đoán thuộc tính (Gender, Hat, Glasses, Backpack)
    par_result = par_engine.predict(crop)

    # 2. Đoán màu sắc (Upper, Lower)
    color_result = color_engine.detect_colors(crop)

    # 3. Hợp nhất kết quả đầy đủ
    full_attributes = {**par_result, **color_result}

    print("  [✓] Ket qua phan tich thuoc tinh toan dien:")
    print("  " + "─" * 45)
    print(f"    • Gioi tinh (Gender)   : {full_attributes['gender']} (Confidence: {full_attributes['gender_confidence']:.2f})")
    print(f"    • Doi mu (Hat)         : {'Co (Yes)' if full_attributes['hat'] else 'Khong (No)'} (Conf: {full_attributes['hat_confidence']:.2f})")
    print(f"    • Deo kinh (Glasses)   : {'Co (Yes)' if full_attributes['glasses'] else 'Khong (No)'} (Conf: {full_attributes['glasses_confidence']:.2f})")
    print(f"    • Deo balo (Backpack)  : {'Co (Yes)' if full_attributes['backpack'] else 'Khong (No)'} (Conf: {full_attributes['backpack_confidence']:.2f})")
    print(f"    • Mau ao (Upper Color) : {full_attributes['upper_color']}")
    print(f"    • Mau quan (Lower)     : {full_attributes['lower_color']}")
    print("  " + "─" * 45)

    return True


def main():
    parser = argparse.ArgumentParser(description="Test Person Attribute Recognition (Phase 5)")
    parser.add_argument("--image", type=str, default=None, help="Path toi anh crop nguoi")
    args = parser.parse_args()

    print("=" * 60)
    print("  PHASE 5 CHECKPOINT: PERSON ATTRIBUTE RECOGNITION (PAR)")
    print("=" * 60)

    ok = test_attribute_recognition(args.image)

    print("\n" + "=" * 60)
    if ok:
        print("  [PASSED] PHASE 5 CHECKPOINT: HOAN THANH!")
        print("  -> San sang chuyen sang PHASE 6: Matching Engine & Search Scoring!")
    else:
        print("  [FAILED] PHASE 5 CHECKPOINT: Co loi xay ra.")
    print("=" * 60)


if __name__ == "__main__":
    main()
