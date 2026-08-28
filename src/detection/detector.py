"""
src/detection/detector.py
=========================
Module phát hiện đối tượng người (Person Detection) sử dụng YOLOv8.

LUỒNG DỮ LIỆU:
    Frame (BGR numpy array)
         ↓
    PersonDetector.detect(frame)
         ↓
    YOLOv8 Inference (lọc class 0 = 'person')
         ↓
    Danh sách Detections:
    [
        {
            "bbox": [x1, y1, x2, y2],   # Tọa độ pixel (int)
            "confidence": 0.89,          # Độ tin cậy (float 0.0 - 1.0)
            "class_id": 0,               # Class ID trong COCO (0 = person)
            "class_name": "person"       # Tên class
        },
        ...
    ]
"""

import os
import torch
import numpy as np
from ultralytics import YOLO
from src.utils.logger import get_logger

logger = get_logger("detector")


class PersonDetector:
    """
    Wrapper chuyên biệt cho YOLOv8 nhằm phát hiện người (class=0) trong frame ảnh/video.
    """

    def __init__(
        self,
        model_path: str = "models/yolo/yolov8n.pt",
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.5,
        device: str = None
    ):
        """
        Khởi tạo mô hình YOLO Person Detector.

        INPUT:
            model_path (str): Đường dẫn file weights hoặc tên model ("yolov8n.pt")
            confidence_threshold (float): Ngưỡng tin cậy tối thiểu (mặc định 0.4)
            iou_threshold (float): Ngưỡng IoU cho Non-Maximum Suppression (NMS)
            device (str): "cuda", "cpu" hoặc None (tự động phát hiện GPU/CPU)
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        # Tự động xác định device nếu không chỉ định
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Khởi tạo YOLO Detector: model='{model_path}', device='{self.device}', conf_thresh={self.confidence_threshold}")

        # Tải mô hình YOLO (nếu file .pt chưa có sẵn, Ultralytics sẽ tự động tải về)
        try:
            self.model = YOLO(model_path)
            logger.info("Tải trọng số YOLO thành công!")
        except Exception as e:
            logger.error(f"Lỗi khi tải YOLO weights từ '{model_path}': {e}")
            raise e

    def detect(self, frame: np.ndarray) -> list:
        """
        Phát hiện tất cả người xuất hiện trong 1 frame ảnh.

        INPUT:
            frame (np.ndarray): Ảnh đầu vào dạng numpy array định dạng BGR (H, W, 3)

        OUTPUT:
            detections (list of dict): Danh sách người được phát hiện.
                Mỗi phần tử là dict:
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float,
                    "class_id": 0,
                    "class_name": "person"
                }
        """
        if frame is None or frame.size == 0:
            return []

        # Chạy inference với YOLO:
        # - classes=[0]: CHỈ detect class 0 ('person' trong bộ dữ liệu COCO 80 classes)
        # - conf: lọc bỏ các dự đoán dưới ngưỡng
        # - iou: lọc chồng lấn bounding box
        # - verbose=False: tắt in log thừa từng frame ra terminal
        results = self.model.predict(
            source=frame,
            classes=[0],
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )

        detections = []
        if len(results) == 0:
            return detections

        first_result = results[0]
        boxes = first_result.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        # Trích xuất tọa độ bounding box và confidence
        xyxy_tensor = boxes.xyxy.cpu().numpy()  # Mảng (N, 4): x1, y1, x2, y2
        conf_tensor = boxes.conf.cpu().numpy()  # Mảng (N,): confidence scores

        for bbox, conf in zip(xyxy_tensor, conf_tensor):
            x1, y1, x2, y2 = [int(round(coord)) for coord in bbox]

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(conf),
                "class_id": 0,
                "class_name": "person"
            })

        return detections

    def draw_detections(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """
        Vẽ bounding box và confidence score của các phát hiện lên frame.

        INPUT:
            frame (np.ndarray): Ảnh gốc (sẽ tạo bản copy hoặc vẽ trực tiếp)
            detections (list): Danh sách dict từ hàm detect()

        OUTPUT:
            annotated_frame (np.ndarray): Ảnh đã vẽ bounding boxes
        """
        import cv2

        annotated_frame = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]

            # Màu xanh Cyan cho Person Detection
            box_color = (255, 200, 0) # BGR
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)

            label = f"Person {conf * 100:.1f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            # Nền chữ
            cv2.rectangle(
                annotated_frame,
                (x1, max(0, y1 - text_h - 6)),
                (x1 + text_w + 4, max(0, y1)),
                box_color,
                -1
            )
            # Chữ màu đen
            cv2.putText(
                annotated_frame,
                label,
                (x1 + 2, max(text_h + 2, y1 - 3)),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA
            )

        return annotated_frame
