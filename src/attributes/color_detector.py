"""
Đề xuất cải tiến cho src/attributes/color_detector.py
======================================================
Thay đổi chính so với bản gốc:
1. Chuẩn hóa ánh sáng (gray-world) trên ROI trước khi chuyển sang HSV, giảm phụ
   thuộc vào điều kiện đèn của từng camera/scene.
2. Thay "trimmed median" bằng phân cụm nhẹ (K=2, dùng cv2.kmeans trên không gian
   H-S sau khi loại các pixel quá tối/quá sáng) rồi chọn cụm chiếm số pixel lớn
   nhất -> xử lý tốt hơn với áo có 2 tông màu / có bóng đổ, vẫn rất rẻ (ROI nhỏ).
3. Trả kèm "confidence" (tỉ lệ pixel thuộc cụm thắng) để pipeline/matcher có thể
   hạ trọng số khi màu không rõ ràng (áo họa tiết phức tạp).
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
        """Cân bằng trắng đơn giản kiểu Gray-World để giảm lệch màu do ánh sáng."""
        img = bgr.astype(np.float32)
        mean_b, mean_g, mean_r = img[..., 0].mean(), img[..., 1].mean(), img[..., 2].mean()
        mean_gray = (mean_b + mean_g + mean_r) / 3.0
        eps = 1e-6
        img[..., 0] *= (mean_gray / (mean_b + eps))
        img[..., 1] *= (mean_gray / (mean_g + eps))
        img[..., 2] *= (mean_gray / (mean_r + eps))
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

        # Loại bỏ pixel cực tối/cực sáng (thường là bóng/viền/phản sáng, không mang
        # thông tin màu áo thật) trước khi phân cụm, giữ lại median làm fallback
        v_channel = pixels[:, 2]
        mask = (v_channel > 15) & (v_channel < 250)
        filtered = pixels[mask] if mask.sum() >= 10 else pixels

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

    # ---------- HSV -> tên màu (giữ nguyên logic rule-based gốc) ----------
    def _hsv_to_color_name(self, h: float, s: float, v: float, is_upper: bool = True) -> str:
        if v < 75 or (v < 92 and s < 80):
            return "Black"
        if s < 45 and v >= 170:
            return "White"
        if s < 50 and 75 <= v < 170:
            return "Gray"
        if h <= 6 or h >= 168:
            if s >= 115 and v >= 55:
                return "Red" if is_upper else "Other"
            elif v < 85:
                return "Black"
        if 7 <= h <= 25:
            if s >= 130 and v >= 130:
                return "Yellow" if is_upper else "Other"
            elif v < 90:
                return "Black"
            elif s < 60:
                return "Gray"
            else:
                return "Other"
        if 26 <= h <= 35:
            if s >= 75 and v >= 110:
                return "Yellow" if is_upper else "Other"
            elif v < 85:
                return "Black"
            else:
                return "Gray"
        if 36 <= h <= 84:
            if s >= 40 and v >= 45:
                return "Green" if is_upper else "Other"
            elif v < 80:
                return "Black"
        if 85 <= h <= 135:
            if s >= 35 and v >= 45:
                return "Blue"
            elif v < 75:
                return "Black"
        if 136 <= h <= 167:
            if s >= 45 and v >= 50:
                return "Purple" if is_upper else "Other"
            elif v < 80:
                return "Black"
        if v < 85:
            return "Black"
        elif s < 60:
            return "Gray"
        return "Other"

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
        return self._hsv_to_color_name(h_val, s_val, v_val, is_upper=True), conf

    def get_lower_color(self, person_crop: np.ndarray) -> tuple:
        if person_crop is None or person_crop.size == 0:
            return "Other", 0.0
        roi = self._region(person_crop, upper=False)
        if roi.size == 0:
            return "Other", 0.0
        h_val, s_val, v_val, conf = self._extract_dominant_hsv(roi)
        return self._hsv_to_color_name(h_val, s_val, v_val, is_upper=False), conf

    def detect_colors(self, person_crop: np.ndarray) -> dict:
        upper, upper_conf = self.get_upper_color(person_crop)
        lower, lower_conf = self.get_lower_color(person_crop)
        return {
            "upper_color": upper,
            "upper_color_confidence": round(upper_conf, 3),
            "lower_color": lower,
            "lower_color_confidence": round(lower_conf, 3),
        }
