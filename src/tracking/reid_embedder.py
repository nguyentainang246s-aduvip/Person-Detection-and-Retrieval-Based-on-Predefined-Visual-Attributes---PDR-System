"""
Module trích xuất đặc trưng ngoại hình người (Person Re-Identification Embedding)
Thuộc Giai đoạn 1 của lộ trình nâng cấp PDR-System.

Nhiệm vụ:
  - Nhận ảnh crop người từ Detector/Tracker
  - Trích xuất vector embedding 512 chiều (L2-normalized)
  - Đo độ tương đồng diện mạo qua Cosine Similarity
  - Hỗ trợ kế thừa ID và kết nối lại các track bị đứt do vật cản (occlusion)
"""

import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("reid_embedder")


class ReIDEmbedder:
    """
    Bộ trích xuất vector đặc trưng ngoại hình (Appearance Feature Embedder)
    sử dụng kiến trúc nhẹ (~1.5M tham số) tối ưu cho thời gian thực trên CPU/GPU.
    """

    def __init__(self, backbone: str = "mobilenet_v3", device: str = None, weights_path: str = "models/reid/reid_mobilenetv3.pth"):
        self.device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.device_str)
        self.backbone_name = backbone
        self.feature_dim = 512
        self.weights_path = weights_path

        logger.info(f"Đang khởi tạo Re-ID Embedder (backbone='{backbone}', device='{self.device_str}')...")

        if backbone == "mobilenet_v3":
            base_model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
            self.features = base_model.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            # Chiếu từ 576 về vector không gian Re-ID chuẩn 512 chiều
            self.proj = nn.Sequential(
                nn.Linear(576, self.feature_dim),
                nn.BatchNorm1d(self.feature_dim)
            )
        else:
            # Fallback về MobileNetV3
            logger.warning(f"Backbone '{backbone}' chưa có trọng số local, sử dụng mobilenet_v3 mặc định.")
            base_model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
            self.features = base_model.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.proj = nn.Sequential(
                nn.Linear(576, self.feature_dim),
                nn.BatchNorm1d(self.feature_dim)
            )

        # Nạp weights đã fine-tune trên Market-1501 nếu có
        if self.weights_path and os.path.exists(self.weights_path):
            try:
                ckpt = torch.load(self.weights_path, map_location=self.device, weights_only=False)
                sd = ckpt.get("model_state_dict", ckpt)
                self.load_state_dict(sd)
                logger.info(f"Nạp trọng số Re-ID đã fine-tune từ '{self.weights_path}' thành công!")
            except Exception as e:
                logger.warning(f"Không thể nạp checkpoint Re-ID ({e}), dùng ImageNet pretrained.")

        self.features.to(self.device).eval()
        self.pool.to(self.device).eval()
        self.proj.to(self.device).eval()

        # Tiền xử lý chuẩn Re-ID (Kích thước 256x128 chuẩn tập Market-1501/DukeMTMC)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        logger.info(f"Re-ID Embedder sẵn sàng hoạt động (Output dimension: {self.feature_dim}-D L2-normalized)")

    def state_dict(self) -> dict:
        """Trả về toàn bộ state_dict của features và projection head."""
        sd = {}
        for k, v in self.features.state_dict().items():
            sd[f"features.{k}"] = v
        for k, v in self.proj.state_dict().items():
            sd[f"proj.{k}"] = v
        return sd

    def load_state_dict(self, state_dict: dict, strict: bool = False):
        """Nạp state_dict vào features và projection head."""
        feat_sd = {}
        proj_sd = {}
        for k, v in state_dict.items():
            if k.startswith("features."):
                feat_sd[k[len("features."):]] = v
            elif k.startswith("proj."):
                proj_sd[k[len("proj."):]] = v
            elif "classifier" in k:
                continue
            else:
                feat_sd[k] = v

        if feat_sd:
            self.features.load_state_dict(feat_sd, strict=strict)
        if proj_sd:
            self.proj.load_state_dict(proj_sd, strict=strict)

    @torch.no_grad()
    def extract(self, person_crop: np.ndarray) -> np.ndarray:
        """
        Trích xuất vector embedding từ 1 ảnh crop người.

        Args:
            person_crop: BGR/RGB numpy array của người.

        Returns:
            np.ndarray: Vector 512-dim đã L2-normalize.
        """
        if person_crop is None or person_crop.size == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)

        h, w = person_crop.shape[:2]
        if h < 15 or w < 10:
            # Crop quá nhỏ không đủ thông tin ngoại hình
            return np.zeros(self.feature_dim, dtype=np.float32)

        # Chuyển BGR sang RGB nếu cần
        if len(person_crop.shape) == 3 and person_crop.shape[2] == 3:
            rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        else:
            rgb_crop = person_crop

        try:
            tensor = self.transform(rgb_crop).unsqueeze(0).to(self.device)
            feat_map = self.features(tensor)
            pooled = self.pool(feat_map).flatten(1)
            emb = self.proj(pooled)

            # L2-normalization chuẩn metric learning: ||v|| = 1.0
            norm = torch.norm(emb, p=2, dim=1, keepdim=True).clamp(min=1e-6)
            norm_emb = (emb / norm).cpu().numpy().squeeze(0)
            return norm_emb
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất embedding: {e}")
            return np.zeros(self.feature_dim, dtype=np.float32)

    @torch.no_grad()
    def extract_batch(self, person_crops: list) -> list:
        """
        Trích xuất embedding cho danh sách crops theo batch thực sự.
        Stack tất cả tensor hợp lệ -> 1 forward pass duy nhất.
        """
        if not person_crops:
            return []

        valid_indices = []
        tensors = []
        results = [np.zeros(self.feature_dim, dtype=np.float32) for _ in range(len(person_crops))]

        for i, crop in enumerate(person_crops):
            if crop is None or crop.size == 0:
                continue
            h, w = crop.shape[:2]
            if h < 15 or w < 10:
                continue

            if len(crop.shape) == 3 and crop.shape[2] == 3:
                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            else:
                rgb_crop = crop

            try:
                tensors.append(self.transform(rgb_crop))
                valid_indices.append(i)
            except Exception:
                continue

        if not tensors:
            return results

        # Stack thành 1 batch tensor -> 1 forward pass
        batch_tensor = torch.stack(tensors).to(self.device)
        feat_maps = self.features(batch_tensor)
        pooled = self.pool(feat_maps).flatten(1)
        embeddings = self.proj(pooled)

        # L2-normalize
        norms = torch.norm(embeddings, p=2, dim=1, keepdim=True).clamp(min=1e-6)
        norm_embeddings = (embeddings / norms).cpu().numpy()

        for batch_idx, orig_idx in enumerate(valid_indices):
            results[orig_idx] = norm_embeddings[batch_idx]

        return results

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Tính Cosine Similarity giữa 2 vector embedding:
        S = (u . v) / (||u|| * ||v||)
        Vì cả 2 vector đã được L2-normalize nên Cosine Similarity chính là tích vô hướng dot product.
        """
        if emb1 is None or emb2 is None:
            return 0.0
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 < 1e-5 or norm2 < 1e-5:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    @staticmethod
    def update_moving_average(old_emb: np.ndarray, new_emb: np.ndarray, alpha: float = 0.8) -> np.ndarray:
        """
        Cập nhật mượt mà embedding của một track theo thời gian:
        v_smooth = alpha * v_old + (1 - alpha) * v_new
        """
        if old_emb is None or np.all(old_emb == 0):
            return new_emb
        if new_emb is None or np.all(new_emb == 0):
            return old_emb
        smooth = alpha * old_emb + (1.0 - alpha) * new_emb
        norm = np.linalg.norm(smooth)
        if norm > 1e-6:
            smooth = smooth / norm
        return smooth
