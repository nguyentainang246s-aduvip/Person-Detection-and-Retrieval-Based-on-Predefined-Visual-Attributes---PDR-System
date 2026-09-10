"""
src/attributes/learned_color_head.py
====================================
Module nhận dạng màu sắc quần áo bằng Deep Learning (Giai đoạn 2 của lộ trình nâng cấp).

Khắc phục triệt để các nhược điểm của phương pháp truyền thống (K-Means HSV):
  - Kháng bóng đổ và ngược sáng mạnh nhờ Data Augmentation (Synthetic Shadow, ColorJitter).
  - Nhận diện chính xác quần áo có hoa văn, họa tiết phức tạp.
  - Phân loại trực tiếp sang 8 nhóm màu chuẩn:
    [Black, White, Gray, Red, Blue, Yellow, Green, Other]
"""

import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("learned_color_head")

COLOR_CLASSES = ["Black", "White", "Gray", "Red", "Blue", "Yellow", "Green", "Other"]
N_COLORS = len(COLOR_CLASSES)


def add_synthetic_shadow(img: Image.Image) -> Image.Image:
    """
    Tạo gradient bóng đổ giả lập (Synthetic Shadow) ngẫu nhiên lên ảnh
    để huấn luyện mạng nơ-ron kháng ngược sáng và bóng râm ngoài trời.
    """
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]
    alpha = float(np.random.uniform(0.35, 0.65))
    x1, y1 = np.random.randint(0, max(1, w // 2)), 0
    x2, y2 = np.random.randint(max(1, w // 2), w), h

    mask = np.ones((h, w, 1), dtype=np.float32)
    poly = np.array([[0, 0], [x1, y1], [x2, y2], [0, h]], dtype=np.int32)
    cv2.fillPoly(mask, [poly], alpha)
    arr = arr * mask
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


class DualColorNet(nn.Module):
    """
    Kiến trúc mạng nơ-ron tích chập đa nhiệm phân loại màu sắc áo (Upper) và quần (Lower).
    Sử dụng backbone MobileNetV3-Small (~1.5M tham số) để đảm bảo độ trễ cực thấp (< 5ms).
    """

    def __init__(self, n_classes: int = N_COLORS, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base = mobilenet_v3_small(weights=weights)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Nhánh phân loại màu áo (Upper Color Head)
        self.upper_head = nn.Sequential(
            nn.Linear(576, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, n_classes)
        )

        # Nhánh phân loại màu quần (Lower Color Head)
        self.lower_head = nn.Sequential(
            nn.Linear(576, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x: torch.Tensor):
        feat = self.features(x)
        pooled = self.pool(feat).flatten(1)
        upper_logits = self.upper_head(pooled)
        lower_logits = self.lower_head(pooled)
        return upper_logits, lower_logits


class LearnedColorDetector:
    """
    Engine dự đoán màu sắc bằng Deep Learning Head.
    """

    def __init__(self, weights_path: str = "models/par/color_head.pth", device: str = None):
        self.device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.device_str)
        self.weights_path = weights_path
        self.classes = COLOR_CLASSES

        self.model = DualColorNet(n_classes=len(self.classes), pretrained=True)

        if os.path.exists(weights_path):
            try:
                ckpt = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(ckpt.get("model_state_dict", ckpt))
                logger.info(f"Đã nạp trọng số Deep Learning Color Head từ: {weights_path}")
            except Exception as e:
                logger.warning(f"Không thể nạp trọng số từ {weights_path} ({e}), sử dụng pretrained backbone.")
        else:
            logger.info(f"Chưa tìm thấy checkpoint '{weights_path}', khởi tạo với ImageNet feature backbone.")

        self.model.to(self.device).eval()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 112)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def detect_colors(self, person_crop: np.ndarray) -> dict:
        """
        Dự đoán màu sắc áo và quần từ ảnh crop người.

        Returns:
            dict: {
                'upper_color': str,
                'upper_color_confidence': float,
                'lower_color': str,
                'lower_color_confidence': float
            }
        """
        if person_crop is None or person_crop.size == 0 or person_crop.shape[0] < 20 or person_crop.shape[1] < 10:
            return {
                "upper_color": "Other",
                "upper_color_confidence": 0.0,
                "lower_color": "Other",
                "lower_color_confidence": 0.0,
            }

        # BGR -> RGB
        if len(person_crop.shape) == 3 and person_crop.shape[2] == 3:
            rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        else:
            rgb_crop = person_crop

        tensor = self.transform(rgb_crop).unsqueeze(0).to(self.device)
        upper_logits, lower_logits = self.model(tensor)

        upper_probs = torch.softmax(upper_logits, dim=1).cpu().numpy()[0]
        lower_probs = torch.softmax(lower_logits, dim=1).cpu().numpy()[0]

        upper_idx = int(np.argmax(upper_probs))
        lower_idx = int(np.argmax(lower_probs))

        return {
            "upper_color": self.classes[upper_idx],
            "upper_color_confidence": round(float(upper_probs[upper_idx]), 3),
            "lower_color": self.classes[lower_idx],
            "lower_color_confidence": round(float(lower_probs[lower_idx]), 3)
        }

    @torch.no_grad()
    def detect_colors_batch(self, person_crops: list) -> list:
        """Batch inference cho nhiều crops cùng lúc (D3) — tối ưu tốc độ CPU/GPU."""
        results = []
        valid_indices = []
        tensors = []

        default_result = {
            "upper_color": "Other", "upper_color_confidence": 0.0,
            "lower_color": "Other", "lower_color_confidence": 0.0,
        }

        for i, crop in enumerate(person_crops):
            if crop is None or crop.size == 0 or crop.shape[0] < 20:
                results.append(default_result.copy())
                continue

            results.append(None)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if (len(crop.shape) == 3 and crop.shape[2] == 3) else crop
            try:
                tensors.append(self.transform(rgb))
                valid_indices.append(i)
            except Exception:
                results[i] = default_result.copy()

        if tensors:
            batch = torch.stack(tensors).to(self.device)
            upper_logits, lower_logits = self.model(batch)
            upper_probs = torch.softmax(upper_logits, dim=1).cpu().numpy()
            lower_probs = torch.softmax(lower_logits, dim=1).cpu().numpy()

            for batch_idx, orig_idx in enumerate(valid_indices):
                u_idx = int(np.argmax(upper_probs[batch_idx]))
                l_idx = int(np.argmax(lower_probs[batch_idx]))
                results[orig_idx] = {
                    "upper_color": self.classes[u_idx],
                    "upper_color_confidence": round(float(upper_probs[batch_idx][u_idx]), 3),
                    "lower_color": self.classes[l_idx],
                    "lower_color_confidence": round(float(lower_probs[batch_idx][l_idx]), 3),
                }

        for i in range(len(results)):
            if results[i] is None:
                results[i] = default_result.copy()

        return results
