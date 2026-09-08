"""
src/attributes/par_model.py
===========================
Module nhận dạng các thuộc tính người (PAR - Pedestrian Attribute Recognition)
sử dụng mạng ResNet50 Multi-label Classification Head.

CÁC THUỘC TÍNH (4 THUỘC TÍNH CHÍNH):
    0: Giới tính (Female / Male)
    1: Đội mũ (Hat)
    2: Đeo kính (Glasses)
    3: Đeo balo (Backpack)
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("par_model")


class ResNet50PAR(nn.Module):
    """
    Kiến trúc ResNet50 fine-tune cho Multi-label Classification.
    """

    def __init__(self, n_attrs: int = 4, pretrained: bool = True):
        super().__init__()
        if pretrained:
            base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            base_model = models.resnet50(weights=None)

        # Trích xuất backbone ResNet50 (bỏ lớp FC cuối cùng)
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])

        # Head phân loại đa nhãn
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, n_attrs),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        probs = self.classifier(feat)
        return probs


class AttributeRecognizer:
    """
    Engine dự đoán thuộc tính ngoại hình từ ảnh Person Crop.
    """

    # Danh sách 4 thuộc tính chính
    ATTR_NAMES = ["female", "hat", "glasses", "backpack"]

    def __init__(
        self,
        weights_path: str = "models/par/par_resnet50.pth",
        device: str = None,
        thresholds: dict = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_path = weights_path

        # Ngưỡng quyết định đã được cân chỉnh tối ưu chống nhiễu
        self.thresholds = thresholds or {
            "gender": 0.50,
            "hat": 0.62,       # Nâng nhẹ ngưỡng Mũ để tránh bắt nhầm tóc đen/búi tóc
            "glasses": 0.50,
            "backpack": 0.50
        }

        # Pipeline tiền xử lý ảnh chuẩn cho mạng ResNet50 (224x112 - tỷ lệ cơ thể người)
        self.transform = T.Compose([
            T.Resize((224, 112)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.model = ResNet50PAR(n_attrs=len(self.ATTR_NAMES), pretrained=True)

        # Nạp trọng số nếu file tồn tại
        if os.path.exists(weights_path):
            try:
                try:
                    checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
                except TypeError:
                    checkpoint = torch.load(weights_path, map_location=self.device)

                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint

                # Tương thích tên biến giữa head và classifier
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("head."):
                        new_key = k.replace("head.", "classifier.")
                        new_state_dict[new_key] = v
                    elif k.startswith("classifier.") and not hasattr(self.model, "classifier"):
                        new_key = k.replace("classifier.", "head.")
                        new_state_dict[new_key] = v
                    else:
                        new_state_dict[k] = v

                self.model.load_state_dict(new_state_dict)
                logger.info(f"Nạp trọng số PAR đã fine-tune từ '{weights_path}' thành công!")
            except Exception as e:
                logger.warning(f"Không thể nạp checkpoint: {e}. Sử dụng pretrained ImageNet backbone.")
        else:
            logger.info(f"Chưa tìm thấy '{weights_path}'. Sử dụng backbone ImageNet (chạy ở chế độ Heuristic/Demo).")

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, person_crop: np.ndarray) -> dict:
        """
        Dự đoán các thuộc tính ngoại hình từ ảnh crop người.
        """
        if person_crop is None or person_crop.size == 0:
            return {
                "gender": "Male", "gender_confidence": 0.5,
                "hat": False, "hat_confidence": 0.0,
                "glasses": False, "glasses_confidence": 0.0,
                "backpack": False, "backpack_confidence": 0.0,
                "raw_probs": {"female": 0.0, "hat": 0.0, "glasses": 0.0, "backpack": 0.0}
            }

        # BGR (OpenCV) -> RGB (PIL)
        rgb_img = Image.fromarray(person_crop[:, :, ::-1])
        tensor = self.transform(rgb_img).unsqueeze(0).to(self.device)

        # Forward pass
        probs = self.model(tensor).squeeze(0).cpu().numpy()

        p_female = float(probs[0])
        p_hat = float(probs[1])
        p_glasses = float(probs[2])
        p_backpack = float(probs[3])

        is_female = p_female >= self.thresholds["gender"]
        gender_label = "Female" if is_female else "Male"
        gender_conf = p_female if is_female else (1.0 - p_female)

        return {
            "gender": gender_label,
            "gender_confidence": round(gender_conf, 3),
            "hat": p_hat >= self.thresholds["hat"],
            "hat_confidence": round(p_hat, 3),
            "glasses": p_glasses >= self.thresholds["glasses"],
            "glasses_confidence": round(p_glasses, 3),
            "backpack": p_backpack >= self.thresholds["backpack"],
            "backpack_confidence": round(p_backpack, 3),
            "raw_probs": {
                "female": p_female,
                "hat": p_hat,
                "glasses": p_glasses,
                "backpack": p_backpack
            }
        }

    def _empty_result(self) -> dict:
        return {
            "gender": "Male", "gender_confidence": 0.5,
            "hat": False, "hat_confidence": 0.0,
            "glasses": False, "glasses_confidence": 0.0,
            "backpack": False, "backpack_confidence": 0.0,
            "raw_probs": {"female": 0.0, "hat": 0.0, "glasses": 0.0, "backpack": 0.0}
        }

    @torch.no_grad()
    def predict_batch(self, person_crops: list) -> list:
        valid_idx, tensors = [], []
        for i, crop in enumerate(person_crops):
            if crop is not None and crop.size > 0:
                rgb = Image.fromarray(crop[:, :, ::-1])
                tensors.append(self.transform(rgb))
                valid_idx.append(i)

        results = [self._empty_result() for _ in person_crops]
        if not tensors:
            return results

        batch = torch.stack(tensors).to(self.device)
        probs = self.model(batch).cpu().numpy()

        for row_idx, orig_idx in enumerate(valid_idx):
            p_female, p_hat, p_glasses, p_backpack = probs[row_idx]
            is_female = p_female >= self.thresholds["gender"]
            results[orig_idx] = {
                "gender": "Female" if is_female else "Male",
                "gender_confidence": round(float(p_female if is_female else 1 - p_female), 3),
                "hat": bool(p_hat >= self.thresholds["hat"]),
                "hat_confidence": round(float(p_hat), 3),
                "glasses": bool(p_glasses >= self.thresholds["glasses"]),
                "glasses_confidence": round(float(p_glasses), 3),
                "backpack": bool(p_backpack >= self.thresholds["backpack"]),
                "backpack_confidence": round(float(p_backpack), 3),
                "raw_probs": {
                    "female": float(p_female), "hat": float(p_hat),
                    "glasses": float(p_glasses), "backpack": float(p_backpack),
                },
            }
        return results
