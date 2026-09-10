"""
src/retrieval/result_writer.py
==============================
Ghi kết quả tìm kiếm ra CSV, ảnh crop, và quản lý xuất file kết quả.
Tách từ pipeline.py nhằm chuyên biệt hóa tác vụ I/O.
"""

import os
import csv
import cv2
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger("result_writer")


class ResultWriter:
    """Ghi kết quả ra file CSV và lưu ảnh crop."""

    def __init__(self, csv_path: Optional[str] = None, crops_dir: str = "results/crops"):
        self.csv_path = csv_path
        self.crops_dir = crops_dir
        self.csv_file = None
        self.csv_writer = None

        os.makedirs(crops_dir, exist_ok=True)

        if csv_path:
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            self.csv_file = open(csv_path, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                "timestamp", "frame_idx", "track_id", "matching_score",
                "gender", "upper_color", "lower_color",
                "hat", "glasses", "backpack", "crop_image_path"
            ])

    def save_crop(self, crop, track_id: int, timestamp: str) -> str:
        """Lưu ảnh crop, trả về đường dẫn file."""
        crop_filename = f"target_track_{track_id:03d}_{timestamp.replace(':', '-')}.jpg"
        crop_path = os.path.join(self.crops_dir, crop_filename)
        if crop is not None and crop.size > 0:
            cv2.imwrite(crop_path, crop)
        return crop_path

    def write_csv_row(self, timestamp: str, frame_idx: int, track_id: int, score: float, attrs: dict, crop_path: str):
        if self.csv_writer:
            self.csv_writer.writerow([
                timestamp, frame_idx, track_id, f"{score*100:.1f}%",
                attrs.get("gender"), attrs.get("upper_color"),
                attrs.get("lower_color"), attrs.get("hat"),
                attrs.get("glasses"), attrs.get("backpack"), crop_path
            ])
            self.csv_file.flush()

    def close(self):
        if self.csv_file:
            try:
                self.csv_file.close()
                self.csv_file = None
            except Exception as e:
                logger.warning(f"Lỗi khi đóng file CSV: {e}")
