"""
src/tracking/tracker.py
=======================
Module theo dõi đa đối tượng (Multi-Object Tracking - MOT) sử dụng ByteTrack.

LUỒNG DỮ LIỆU:
    Frame t (BGR numpy array)
         ↓
    PersonTracker.track(frame)
         ↓
    ByteTrack Association (Kalman Filter + Hungarian Algorithm)
         ↓
    Danh sách Tracked Objects:
    [
        {
            "track_id": 12,              # ID định danh duy nhất không đổi qua các frame
            "bbox": [x1, y1, x2, y2],   # Tọa độ bounding box tại frame hiện tại
            "confidence": 0.89,          # Độ tin cậy
            "class_id": 0,               # Class person
            "class_name": "person"
        },
        ...
    ]
"""

import os
import torch
import numpy as np
from ultralytics import YOLO
from src.utils.logger import get_logger

logger = get_logger("tracker")


class PersonTracker:
    """
    Wrapper tích hợp YOLOv8 + ByteTrack để phát hiện và gán Track ID liên tục cho từng người.
    """

    def __init__(
        self,
        model_path: str = "models/yolo/yolov8n.pt",
        tracker_type: str = "bytetrack.yaml",
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.5,
        device: str = None
    ):
        """
        Khởi tạo Tracker.
        """
        self.model_path = model_path
        self.tracker_type = tracker_type
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Khởi tạo ByteTrack Tracker: model='{model_path}', tracker='{tracker_type}', device='{self.device}'")

        try:
            self.model = YOLO(model_path)
            logger.info("Nạp mô hình YOLO cho Tracker thành công!")
        except Exception as e:
            logger.error(f"Lỗi nạp mô hình: {e}")
            raise e

    def track(self, frame: np.ndarray, persist: bool = True) -> list:
        """
        Chạy tracking trên 1 frame và trả về danh sách người kèm Track ID.
        """
        if frame is None or frame.size == 0:
            return []

        try:
            results = self.model.track(
                source=frame,
                persist=persist,
                tracker=self.tracker_type,
                classes=[0],                  # Chỉ theo dõi người (class 0)
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )
        except Exception as e:
            # Phục hồi an toàn nếu state tracker bị lỗi khi reset
            logger.warning(f"Tracker warning: {e}. Đang tái khởi tạo predictor...")
            self.reset()
            try:
                results = self.model.track(
                    source=frame,
                    persist=persist,
                    tracker=self.tracker_type,
                    classes=[0],
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    device=self.device,
                    verbose=False
                )
            except Exception as final_e:
                logger.error(f"Lỗi tracking: {final_e}")
                return []

        tracked_objects = []
        if not results or len(results) == 0:
            return tracked_objects

        first_res = results[0]
        boxes = first_res.boxes

        if boxes is None or len(boxes) == 0:
            return tracked_objects

        # Nếu chưa có track id (ví dụ frame đầu tiên), fallback sang bounding box cơ bản
        if boxes.id is None:
            xyxy_tensor = boxes.xyxy.cpu().numpy()
            conf_tensor = boxes.conf.cpu().numpy()
            for idx, (bbox, conf) in enumerate(zip(xyxy_tensor, conf_tensor)):
                x1, y1, x2, y2 = [int(round(coord)) for coord in bbox]
                tracked_objects.append({
                    "track_id": idx + 1,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(conf),
                    "class_id": 0,
                    "class_name": "person"
                })
            return tracked_objects

        xyxy_tensor = boxes.xyxy.cpu().numpy()
        conf_tensor = boxes.conf.cpu().numpy()
        id_tensor = boxes.id.int().cpu().numpy()

        for bbox, conf, track_id in zip(xyxy_tensor, conf_tensor, id_tensor):
            x1, y1, x2, y2 = [int(round(coord)) for coord in bbox]

            tracked_objects.append({
                "track_id": int(track_id),
                "bbox": [x1, y1, x2, y2],
                "confidence": float(conf),
                "class_id": 0,
                "class_name": "person"
            })

        return tracked_objects

    def reset(self):
        """
        Reset trạng thái bộ nhớ tracking an toàn (dùng khi chuyển sang video mới hoặc dừng tìm kiếm).
        """
        try:
            # P1-2: Tránh load lại weights từ đĩa gây bottleneck (tiết kiệm hàng trăm ms)
            if hasattr(self.model, 'predictor') and self.model.predictor is not None:
                # Ultralytics sẽ tự tạo lại predictor và tracker mới khi predict ở frame tiếp theo
                self.model.predictor = None
            else:
                self.model = YOLO(self.model_path)
        except Exception:
            try:
                self.model = YOLO(self.model_path)
            except Exception:
                pass
        logger.info("Đã reset trạng thái Tracker thành công.")
