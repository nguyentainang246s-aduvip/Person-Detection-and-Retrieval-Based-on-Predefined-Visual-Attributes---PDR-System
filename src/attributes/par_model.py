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
    Đầu ra là Logits (không dùng Sigmoid ở model để dùng BCEWithLogitsLoss tối ưu số học).
    """

    def __init__(self, n_attrs: int = 4, pretrained: bool = True):
        super().__init__()
        if pretrained:
            base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            base_model = models.resnet50(weights=None)

        # Trích xuất backbone ResNet50 (bỏ lớp FC cuối cùng)
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])

        # Head phân loại đa nhãn (trả về logits thô)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, n_attrs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        logits = self.classifier(feat)
        return logits


class MobileNetV3PAR(nn.Module):
    """
    Kiến trúc MobileNetV3-Large cho PAR (Giai đoạn 3).
    Siêu nhẹ (~5.5M tham số, FLOPs ~0.23 GFLOPs), tăng tốc suy luận trên CPU/Edge.
    """
    def __init__(self, n_attrs: int = 4, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        base_model = models.mobilenet_v3_large(weights=weights)
        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(960, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, n_attrs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        pooled = self.pool(feat)
        return self.classifier(pooled)


class EfficientNetPAR(nn.Module):
    """
    Kiến trúc EfficientNet-B0 cho PAR (Giai đoạn 3).
    Cân bằng hoàn hảo giữa tham số (~5.3M) và độ chính xác đặc trưng đa tỉ lệ.
    """
    def __init__(self, n_attrs: int = 4, pretrained: bool = True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base_model = models.efficientnet_b0(weights=weights)
        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, n_attrs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        pooled = self.pool(feat)
        return self.classifier(pooled)


class FocalLoss(nn.Module):
    """
    Focal Loss cho Multi-label Classification (Lin et al., ICCV 2017).
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Hệ số tập trung (focusing parameter), mặc định 2.0
        alpha: Trọng số class imbalance, có thể là:
               - None: không dùng alpha weighting
               - float: cùng alpha cho tất cả class
               - Tensor shape (n_attrs,): alpha riêng cho từng attribute
    """
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        # Đăng ký alpha là buffer để tự động chuyển device cùng model
        if alpha is not None and isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = alpha  # scalar hoặc None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = (1.0 - p_t) ** self.gamma

        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                # Per-class alpha: broadcast shape (n_attrs,) -> (batch, n_attrs)
                alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            else:
                # Scalar alpha: cùng giá trị cho tất cả class
                alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def build_par_model(backbone: str = "resnet50", n_attrs: int = 4, pretrained: bool = True) -> nn.Module:
    """Factory function khởi tạo PAR model theo backbone mong muốn."""
    b_name = backbone.lower()
    if b_name == "resnet50":
        return ResNet50PAR(n_attrs=n_attrs, pretrained=pretrained)
    elif b_name in ["mobilenet_v3", "mobilenetv3", "mobilenet"]:
        return MobileNetV3PAR(n_attrs=n_attrs, pretrained=pretrained)
    elif b_name in ["efficientnet_b0", "efficientnet", "effnet"]:
        return EfficientNetPAR(n_attrs=n_attrs, pretrained=pretrained)
    else:
        logger.warning(f"Không nhận diện được backbone '{backbone}', fallback về ResNet50.")
        return ResNet50PAR(n_attrs=n_attrs, pretrained=pretrained)


class AttributeRecognizer:
    """
    Engine dự đoán thuộc tính ngoại hình từ ảnh Person Crop.
    Hỗ trợ cả mô hình 4 thuộc tính (PA-100K) và mô hình 40 thuộc tính chuẩn quốc tế UPAR Benchmark.
    """

    # Danh sách 4 thuộc tính chính trong PDR-System
    ATTR_NAMES = ["female", "hat", "glasses", "backpack"]

    # Ánh xạ index chuẩn của UPAR (WACV Benchmark - 40 binary attributes)
    UPAR_ATTR_MAPPING = {
        "female": 0,      # gender_female
        "hat": 38,        # accessory_hat
        "glasses": 37,    # accessory_glasses
        "backpack": 35    # accessory_backpack
    }

    # Ánh xạ index chuẩn của PA-100K (4 binary attributes)
    PA100K_ATTR_MAPPING = {
        "female": 0,
        "hat": 1,
        "glasses": 2,
        "backpack": 3
    }

    def __init__(
        self,
        weights_path: str = "models/par/par_resnet50.pth",
        device: str = None,
        thresholds: dict = None,
        backbone: str = "resnet50",
        onnx_path: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_path = weights_path
        self.backbone = backbone

        # Ngưỡng quyết định tối ưu F1
        self.thresholds = thresholds or {
            "gender": 0.50,
            "hat": 0.62,       # Nâng nhẹ ngưỡng Mũ để tránh bắt nhầm tóc đen/búi tóc
            "glasses": 0.50,
            "backpack": 0.50
        }

        # Pipeline tiền xử lý ảnh chuẩn: Resize(256, 128) -> CenterCrop(224, 112)
        self.transform = T.Compose([
            T.Resize((256, 128)),
            T.CenterCrop((224, 112)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Mặc định PA-100K 4 thuộc tính
        self.n_attrs = len(self.ATTR_NAMES)
        self.attr_indices = dict(self.PA100K_ATTR_MAPPING)
        state_dict_to_load = None

        # Kiểm tra trước checkpoint để xác định số lượng thuộc tính (4 vs 40 UPAR)
        if os.path.exists(weights_path):
            try:
                try:
                    checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
                except TypeError:
                    checkpoint = torch.load(weights_path, map_location=self.device)

                if isinstance(checkpoint, dict):
                    sd = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
                    if "thresholds" in checkpoint and isinstance(checkpoint["thresholds"], dict):
                        self.thresholds.update(checkpoint["thresholds"])
                    if "attr_indices" in checkpoint and isinstance(checkpoint["attr_indices"], dict):
                        self.attr_indices.update(checkpoint["attr_indices"])
                else:
                    sd = checkpoint

                # Phát hiện số class đầu ra từ layer cuối cùng
                for k, v in reversed(list(sd.items())):
                    if ("classifier" in k or "head" in k or "fc" in k) and k.endswith(".weight"):
                        out_dim = v.shape[0]
                        if out_dim == 40:
                            self.n_attrs = 40
                            self.attr_indices = dict(self.UPAR_ATTR_MAPPING)
                            logger.info("Phát hiện checkpoint chuẩn UPAR (40 thuộc tính). Tự động kích hoạt UPAR attribute mapping.")
                        elif out_dim == 4:
                            self.n_attrs = 4
                            self.attr_indices = dict(self.PA100K_ATTR_MAPPING)
                        else:
                            self.n_attrs = out_dim
                        break
                state_dict_to_load = sd
            except Exception as e:
                logger.warning(f"Lỗi kiểm tra cấu trúc checkpoint {weights_path}: {e}")

        self.model = build_par_model(backbone=backbone, n_attrs=self.n_attrs, pretrained=True)

        # ONNX Runtime Fast Path (D1)
        self.onnx_session = None
        resolved_onnx = onnx_path
        if not resolved_onnx:
            default_onnx_int8 = f"models/par/par_{backbone}_int8.onnx"
            default_onnx_fp32 = f"models/par/par_{backbone}.onnx"
            if os.path.exists(default_onnx_int8):
                resolved_onnx = default_onnx_int8
            elif os.path.exists(default_onnx_fp32):
                resolved_onnx = default_onnx_fp32

        if resolved_onnx and os.path.exists(resolved_onnx):
            try:
                from src.utils.onnx_engine import ONNXInferenceEngine
                self.onnx_session = ONNXInferenceEngine(resolved_onnx, device=device)
                logger.info(f"Đã nạp ONNX Runtime engine: {resolved_onnx} ({self.onnx_session.file_size_mb:.2f} MB)")
            except Exception as e:
                logger.warning(f"Không thể khởi tạo ONNX engine ({e}), dùng PyTorch fallback.")
                self.onnx_session = None

        # Nạp trọng số vào model
        if state_dict_to_load is not None:
            try:
                new_state_dict = {}
                for k, v in state_dict_to_load.items():
                    if k.startswith("head."):
                        new_key = k.replace("head.", "classifier.")
                        new_state_dict[new_key] = v
                    elif k.startswith("classifier.") and not hasattr(self.model, "classifier"):
                        new_key = k.replace("classifier.", "head.")
                        new_state_dict[new_key] = v
                    else:
                        new_state_dict[k] = v

                self.model.load_state_dict(new_state_dict, strict=False)
                logger.info(f"Nạp trọng số PAR từ '{weights_path}' thành công! (n_attrs={self.n_attrs})")
            except Exception as e:
                logger.warning(f"Không thể nạp checkpoint: {e}. Sử dụng pretrained ImageNet backbone.")
        else:
            logger.info(f"Chưa tìm thấy '{weights_path}'. Sử dụng backbone ImageNet (chạy ở chế độ Heuristic/Demo).")

        self.model.to(self.device)
        self.model.eval()

    def export_onnx(self, output_path: str = "models/par/par_resnet50.onnx"):
        """Export model sang ONNX format để tối ưu inference (D1)."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dummy_input = torch.randn(1, 3, 224, 112).to(self.device)
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
            opset_version=17,
        )
        logger.info(f"Đã export ONNX model: {output_path}")

    @torch.inference_mode()
    def predict(self, person_crop: np.ndarray) -> dict:
        """
        Dự đoán các thuộc tính ngoại hình từ ảnh crop người.
        Tự động trích xuất các thuộc tính mục tiêu qua bảng ánh xạ index.
        """
        if person_crop is None or person_crop.size == 0:
            return self._empty_result()

        rgb_img = Image.fromarray(person_crop[:, :, ::-1])
        tensor = self.transform(rgb_img).unsqueeze(0).to(self.device)

        logits = self.model(tensor).squeeze(0).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))

        idx_female = self.attr_indices.get("female", 0)
        idx_hat = self.attr_indices.get("hat", 1)
        idx_glasses = self.attr_indices.get("glasses", 2)
        idx_backpack = self.attr_indices.get("backpack", 3)

        p_female = float(probs[idx_female]) if idx_female < len(probs) else 0.5
        p_hat = float(probs[idx_hat]) if idx_hat < len(probs) else 0.0
        p_glasses = float(probs[idx_glasses]) if idx_glasses < len(probs) else 0.0
        p_backpack = float(probs[idx_backpack]) if idx_backpack < len(probs) else 0.0

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

    def resolve_attributes(self, smooth_probs: dict) -> dict:
        """Chuyển đổi xác suất làm mịn (EMA) thành nhãn và độ tin cậy."""
        p_female = float(smooth_probs.get("female", 0.5))
        p_hat = float(smooth_probs.get("hat", 0.0))
        p_glasses = float(smooth_probs.get("glasses", 0.0))
        p_backpack = float(smooth_probs.get("backpack", 0.0))

        is_female = p_female >= self.thresholds["gender"]
        return {
            "gender": "Female" if is_female else "Male",
            "gender_confidence": round(float(p_female if is_female else 1.0 - p_female), 3),
            "hat": bool(p_hat >= self.thresholds["hat"]),
            "hat_confidence": round(p_hat, 3),
            "glasses": bool(p_glasses >= self.thresholds["glasses"]),
            "glasses_confidence": round(p_glasses, 3),
            "backpack": bool(p_backpack >= self.thresholds["backpack"]),
            "backpack_confidence": round(p_backpack, 3),
        }

    @torch.inference_mode()
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

        if self.onnx_session is not None:
            batch_np = torch.stack(tensors).numpy()
            logits = self.onnx_session.run(batch_np)[0]
        else:
            batch = torch.stack(tensors).to(self.device)
            logits = self.model(batch).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))

        idx_female = self.attr_indices.get("female", 0)
        idx_hat = self.attr_indices.get("hat", 1)
        idx_glasses = self.attr_indices.get("glasses", 2)
        idx_backpack = self.attr_indices.get("backpack", 3)

        for row_idx, orig_idx in enumerate(valid_idx):
            row_probs = probs[row_idx]
            p_female = float(row_probs[idx_female]) if idx_female < len(row_probs) else 0.5
            p_hat = float(row_probs[idx_hat]) if idx_hat < len(row_probs) else 0.0
            p_glasses = float(row_probs[idx_glasses]) if idx_glasses < len(row_probs) else 0.0
            p_backpack = float(row_probs[idx_backpack]) if idx_backpack < len(row_probs) else 0.0

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
