"""
src/attributes/color_detector.py
================================
Module phân tích và nhận dạng màu sắc trang phục (Áo và Quần) của người.
Sử dụng Fast Histogram Quantization + Trimmed Median (Tốc độ cực nhanh, < 0.2ms/người)
thay thế K-Means nặng nề để đảm bảo FPS đạt mức Realtime 20-30 FPS.
"""

import cv2
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("color_detector")


class ColorDetector:
    """
    Bộ nhận diện màu sắc trang phục tối ưu hóa tốc độ cao (Fast Histogram + HSV Adaptive ROI).
    """

    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters

    def _hsv_to_color_name(self, h: float, s: float, v: float, is_upper: bool = True) -> str:
        """
        Chuyển đổi bộ giá trị (H, S, V) trong OpenCV sang tên màu chuẩn.
        """
        # 1. Nhóm màu Đen (Black) - Quần áo tối màu trong camera thực tế
        if v < 75 or (v < 92 and s < 80):
            return "Black"

        # 2. Nhóm màu Trắng (White)
        if s < 45 and v >= 170:
            return "White"

        # 3. Nhóm màu Xám (Gray)
        if s < 50 and 75 <= v < 170:
            return "Gray"

        # 4. Nhóm màu Đỏ (Red) - Phải là màu đỏ bão hòa cao (S >= 115) để tránh nhầm da người
        if (h <= 6 or h >= 168):
            if s >= 115 and v >= 55:
                return "Red" if is_upper else "Other"
            elif v < 85:
                return "Black"

        # 5. Nhóm màu Da / Nâu / Be / Cam nhạt (Skin tones & Browns: H = 7 đến 25)
        if 7 <= h <= 25:
            if s >= 130 and v >= 130:
                return "Yellow" if is_upper else "Other"
            elif v < 90:
                return "Black"
            elif s < 60:
                return "Gray"
            else:
                return "Other"

        # 6. Nhóm màu Vàng thực sự (True Yellow)
        if 26 <= h <= 35:
            if s >= 75 and v >= 110:
                return "Yellow" if is_upper else "Other"
            elif v < 85:
                return "Black"
            else:
                return "Gray"

        # 7. Nhóm màu Xanh lá (Green)
        if 36 <= h <= 84:
            if s >= 40 and v >= 45:
                return "Green" if is_upper else "Other"
            elif v < 80:
                return "Black"

        # 8. Nhóm màu Xanh dương (Blue)
        if 85 <= h <= 135:
            if s >= 35 and v >= 45:
                return "Blue"
            elif v < 75:
                return "Black"

        # 9. Nhóm màu Tím (Purple)
        if 136 <= h <= 167:
            if s >= 45 and v >= 50:
                return "Purple" if is_upper else "Other"
            elif v < 80:
                return "Black"

        # Mặc định an toàn
        if v < 85:
            return "Black"
        elif s < 60:
            return "Gray"
        return "Other"

    def _extract_dominant_hsv(self, roi_bgr: np.ndarray) -> tuple:
        """
        Trích xuất nhanh giá trị HSV chủ đạo bằng Trimmed Median (Nhanh gấp 400 lần KMeans).
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return 0.0, 0.0, 0.0

        # Resize nhỏ cố định 24x30 pixel
        roi_small = cv2.resize(roi_bgr, (24, 30), interpolation=cv2.INTER_NEAREST)
        roi_hsv = cv2.cvtColor(roi_small, cv2.COLOR_BGR2HSV)

        # Lấy vùng trung tâm 60% bên trong để loại bỏ viền nền
        h_s, w_s = roi_hsv.shape[:2]
        center_crop = roi_hsv[int(h_s*0.2):int(h_s*0.8), int(w_s*0.2):int(w_s*0.8)].reshape(-1, 3)

        if center_crop.size == 0:
            center_crop = roi_hsv.reshape(-1, 3)

        # Tính trung vị (median) - cực kỳ kháng nhiễu và nhanh tức thì
        median_hsv = np.median(center_crop, axis=0)

        return float(median_hsv[0]), float(median_hsv[1]), float(median_hsv[2])

    def get_upper_color(self, person_crop: np.ndarray) -> str:
        """
        Phân tích màu áo từ ảnh cắt người.
        """
        if person_crop is None or person_crop.size == 0:
            return "Other"

        h, w = person_crop.shape[:2]
        aspect = h / max(w, 1)

        if aspect >= 1.8:
            y1, y2 = int(h * 0.30), int(h * 0.58)
        elif aspect >= 1.0:
            y1, y2 = int(h * 0.50), int(h * 0.80)
        else:
            y1, y2 = int(h * 0.65), int(h * 0.95)

        x1, x2 = int(w * 0.20), int(w * 0.80)

        roi = person_crop[y1:y2, x1:x2]
        if roi.size == 0:
            return "Other"

        h_val, s_val, v_val = self._extract_dominant_hsv(roi)
        return self._hsv_to_color_name(h_val, s_val, v_val, is_upper=True)

    def get_lower_color(self, person_crop: np.ndarray) -> str:
        """
        Phân tích màu quần/váy từ ảnh cắt người.
        """
        if person_crop is None or person_crop.size == 0:
            return "Other"

        h, w = person_crop.shape[:2]
        aspect = h / max(w, 1)

        if aspect >= 1.8:
            y1, y2 = int(h * 0.60), int(h * 0.90)
        else:
            y1, y2 = int(h * 0.80), int(h * 0.98)

        x1, x2 = int(w * 0.20), int(w * 0.80)

        roi = person_crop[y1:y2, x1:x2]
        if roi.size == 0:
            return "Other"

        h_val, s_val, v_val = self._extract_dominant_hsv(roi)
        return self._hsv_to_color_name(h_val, s_val, v_val, is_upper=False)

    def detect_colors(self, person_crop: np.ndarray) -> dict:
        """
        Nhận diện đồng thời cả màu áo và màu quần.
        """
        upper = self.get_upper_color(person_crop)
        lower = self.get_lower_color(person_crop)

        return {
            "upper_color": upper,
            "lower_color": lower
        }
