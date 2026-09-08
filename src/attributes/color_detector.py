"""
src/attributes/color_detector.py
================================
Module trích xuất và phân tích màu sắc của đối tượng (áo, quần) sử dụng thuật toán K-Means Clustering trên không gian màu HSV.

TÍNH NĂNG CHÍNH:
    - Cân bằng trắng (Gray-world normalization) giúp chống nhiễu màu dưới các điều kiện ánh sáng khác nhau.
    - Phân cụm K-Means động để tìm màu chủ đạo, giải quyết được áo có họa tiết hoặc bóng râm.
    - Lọc điểm ảnh có màu da người (Skin Filtering) để không bị nhận nhầm tay, cổ thành màu áo.
"""

import cv2
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("color_detector")


class ColorDetector:
    def __init__(self, n_clusters: int = 2):
        # n_clusters=2 đủ để tách "màu chủ đạo" khỏi "bóng/nền lẫn vào ROI"
        # mà vẫn rất nhanh trên ROI nhỏ (24x30 -> ~720 điểm ảnh)
        self.n_clusters = max(2, n_clusters)

    # ---------- Illumination normalization ----------
    @staticmethod
    def _gray_world_normalize(bgr: np.ndarray) -> np.ndarray:
        """Cân bằng trắng nhẹ có chặn hệ số gain để không triệt tiêu màu áo sặc sỡ."""
        img = bgr.astype(np.float32)
        mean_b = float(img[..., 0].mean())
        mean_g = float(img[..., 1].mean())
        mean_r = float(img[..., 2].mean())
        mean_gray = (mean_b + mean_g + mean_r) / 3.0
        eps = 1e-6
        # Giới hạn gain trong [0.75, 1.35] để tránh triệt tiêu màu sắc đơn sắc thành màu xám
        gain_b = np.clip(mean_gray / (mean_b + eps), 0.75, 1.35)
        gain_g = np.clip(mean_gray / (mean_g + eps), 0.75, 1.35)
        gain_r = np.clip(mean_gray / (mean_r + eps), 0.75, 1.35)
        img[..., 0] *= gain_b
        img[..., 1] *= gain_g
        img[..., 2] *= gain_r
        return np.clip(img, 0, 255).astype(np.uint8)

    def _extract_dominant_hsv(self, roi_bgr: np.ndarray) -> tuple:
        """
        Trích H,S,V chủ đạo bằng phân cụm K-means nhẹ thay vì median đơn thuần.
        Trả về (h, s, v, confidence) — confidence = tỉ lệ pixel thuộc cụm thắng.
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return 0.0, 0.0, 0.0, 0.0

        roi_small = cv2.resize(roi_bgr, (24, 30), interpolation=cv2.INTER_AREA)
        roi_small = self._gray_world_normalize(roi_small)
        roi_hsv = cv2.cvtColor(roi_small, cv2.COLOR_BGR2HSV)

        h_s, w_s = roi_hsv.shape[:2]
        center = roi_hsv[int(h_s * 0.2):int(h_s * 0.8), int(w_s * 0.2):int(w_s * 0.8)]
        pixels = center.reshape(-1, 3).astype(np.float32)
        if pixels.size == 0:
            pixels = roi_hsv.reshape(-1, 3).astype(np.float32)

        # Loại bỏ pixel cực tối/cực sáng (thường là bóng/viền/phản sáng)
        # P1-3: Skin Filtering - Lọc loại bỏ pixel màu da 
        # (Trong OpenCV: H thuộc 0-179. Ngưỡng màu da thường rơi vào H: 0-12 hoặc 170-179, S: 30-150, V > 60)
        h_channel = pixels[:, 0]
        s_channel = pixels[:, 1]
        v_channel = pixels[:, 2]
        
        is_skin = ((h_channel <= 12) | (h_channel >= 170)) & (s_channel >= 30) & (s_channel <= 150) & (v_channel > 60)
        mask = (v_channel > 15) & (v_channel < 250) & (~is_skin)
        
        filtered = pixels[mask] if mask.sum() >= 10 else pixels

        # D1: Nếu vùng ROI có độ bão hòa rất thấp (quần áo Đen/Trắng/Xám),
        # bỏ qua K-Means trên H vì kênh Hue dao động ngẫu nhiên khi S < 40.
        if float(np.median(filtered[:, 1])) < 40:
            v_val = float(np.median(filtered[:, 2]))
            s_val = float(np.median(filtered[:, 1]))
            h_val = 0.0
            conf = 0.85
            return h_val, s_val, v_val, conf

        if len(filtered) < self.n_clusters:
            median_hsv = np.median(filtered, axis=0)
            return float(median_hsv[0]), float(median_hsv[1]), float(median_hsv[2]), 0.5

        # K-means trên (H, S) — bỏ V khỏi không gian phân cụm vì V dễ bị ảnh hưởng
        # bởi bóng đổ trên cùng 1 màu áo, dễ tách sai cụm nếu đưa vào.
        hs = filtered[:, :2]
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.5)
        _, labels, centers = cv2.kmeans(
            hs, self.n_clusters, None, criteria, attempts=2,
            flags=cv2.KMEANS_PP_CENTERS
        )
        labels = labels.flatten()
        counts = np.bincount(labels, minlength=self.n_clusters)
        winner = int(np.argmax(counts))
        confidence = float(counts[winner]) / float(len(labels))

        winner_mask = labels == winner
        # V lấy median của các pixel thuộc cụm thắng (không lấy từ centers vì V bị loại khỏi kmeans)
        v_val = float(np.median(filtered[winner_mask, 2]))
        h_val, s_val = float(centers[winner][0]), float(centers[winner][1])

        return h_val, s_val, v_val, confidence

    # ---------- HSV -> tên màu (đã cân chỉnh theo dữ liệu thực tế) ----------
    def _hsv_to_color_name(self, h: float, s: float, v: float, is_upper: bool = True) -> str:
        # 1. Nhóm màu trung tính (Neutral Colors)
        if v < 45 or (v < 70 and s < 65):
            return "Black"
        if s < 45 and v >= 170:
            return "White"
        if s < 50 and 50 <= v < 170:
            return "Gray"

        # 2. Nhóm màu sặc sỡ (Chromatic Colors)
        if h <= 7 or h >= 166:
            if s >= 70 and v >= 48:
                return "Red" if is_upper else "Other"
            elif v < 65:
                return "Black"

        if 8 <= h <= 25:
            if s >= 90 and v >= 90:
                return "Yellow" if is_upper else "Other"
            elif v < 65:
                return "Black"
            elif s < 50:
                return "Gray"
            else:
                return "Other"

        if 26 <= h <= 35:
            if s >= 60 and v >= 90:
                return "Yellow" if is_upper else "Other"
            elif v < 65:
                return "Black"
            else:
                return "Gray"

        if 36 <= h <= 84:
            if s >= 35 and v >= 45:
                return "Green" if is_upper else "Other"
            elif v < 65:
                return "Black"

        if 85 <= h <= 135:
            if s >= 30 and v >= 45:
                return "Blue"
            elif v < 65:
                return "Black"

        if 136 <= h <= 165:
            if s >= 40 and v >= 45:
                return "Purple" if is_upper else "Other"
            elif v < 65:
                return "Black"

        if v < 70:
            return "Black"
        elif s < 55:
            return "Gray"
        return "Other"

    def _delta_e_name(self, roi_bgr: np.ndarray) -> str:
        """
        D2: Phân loại màu sặc sỡ bằng khoảng cách Delta-E trong không gian màu CIE Lab chuẩn.
        Phản ánh cảm nhận thị giác con người tốt hơn HSV khi phân biệt Red, Blue, Green, Yellow, Purple.
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return "Other"
        small = cv2.resize(roi_bgr, (24, 30))
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
        pixels = lab.reshape(-1, 3)

        # Lấy median Lab của các pixel
        mean_lab = np.median(pixels, axis=0)

        # Tham chiếu Lab trong thang đo OpenCV 8-bit [L: 0-255, a: 0-255, b: 0-255]
        REF_LAB = {
            "Red": np.array([130.0, 180.0, 160.0]),
            "Blue": np.array([80.0, 140.0, 70.0]),
            "Green": np.array([120.0, 85.0, 155.0]),
            "Yellow": np.array([195.0, 115.0, 195.0]),
            "Purple": np.array([95.0, 165.0, 95.0]),
        }
        dists = {name: float(np.linalg.norm(mean_lab - ref)) for name, ref in REF_LAB.items()}
        best_color = min(dists, key=dists.get)
        return best_color if dists[best_color] < 55.0 else "Other"

    def _region(self, person_crop: np.ndarray, upper: bool):
        h, w = person_crop.shape[:2]
        aspect = h / max(w, 1)
        if upper:
            if aspect >= 1.8:
                y1, y2 = int(h * 0.30), int(h * 0.58)
            elif aspect >= 1.0:
                y1, y2 = int(h * 0.50), int(h * 0.80)
            else:
                y1, y2 = int(h * 0.65), int(h * 0.95)
        else:
            if aspect >= 1.8:
                y1, y2 = int(h * 0.60), int(h * 0.90)
            else:
                y1, y2 = int(h * 0.80), int(h * 0.98)
        x1, x2 = int(w * 0.20), int(w * 0.80)
        return person_crop[y1:y2, x1:x2]

    def get_upper_color(self, person_crop: np.ndarray) -> tuple:
        if person_crop is None or person_crop.size == 0:
            return "Other", 0.0
        roi = self._region(person_crop, upper=True)
        if roi.size == 0:
            return "Other", 0.0
        h_val, s_val, v_val, conf = self._extract_dominant_hsv(roi)
        color_name = self._hsv_to_color_name(h_val, s_val, v_val, is_upper=True)
        # Nếu HSV chưa rõ ("Other") nhưng có độ bão hòa, dùng Delta-E trong CIE Lab để nhận diện
        if color_name == "Other" and s_val >= 35:
            delta_e_color = self._delta_e_name(roi)
            if delta_e_color != "Other":
                color_name = delta_e_color
                conf = max(conf, 0.75)
        return color_name, conf

    def get_lower_color(self, person_crop: np.ndarray) -> tuple:
        if person_crop is None or person_crop.size == 0:
            return "Other", 0.0
        roi = self._region(person_crop, upper=False)
        if roi.size == 0:
            return "Other", 0.0
        h_val, s_val, v_val, conf = self._extract_dominant_hsv(roi)
        color_name = self._hsv_to_color_name(h_val, s_val, v_val, is_upper=False)
        # Nếu HSV chưa rõ ("Other") nhưng có độ bão hòa, dùng Delta-E trong CIE Lab
        if color_name == "Other" and s_val >= 35:
            delta_e_color = self._delta_e_name(roi)
            if delta_e_color in ["Blue", "Black"]:
                color_name = delta_e_color
                conf = max(conf, 0.75)
        return color_name, conf

    def detect_colors(self, person_crop: np.ndarray) -> dict:
        upper, upper_conf = self.get_upper_color(person_crop)
        lower, lower_conf = self.get_lower_color(person_crop)
        return {
            "upper_color": upper,
            "upper_color_confidence": round(upper_conf, 3),
            "lower_color": lower,
            "lower_color_confidence": round(lower_conf, 3),
        }
