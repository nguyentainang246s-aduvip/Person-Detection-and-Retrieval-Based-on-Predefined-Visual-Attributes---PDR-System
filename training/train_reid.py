"""
training/train_reid.py
======================
Script thực hiện ƯU TIÊN 4: Huấn luyện / Fine-tune module Re-ID với Market-1501.
Sử dụng phương pháp:
    - Triplet Loss + Online Hard Mining (Hermans et al., 2017)
    - CrossEntropy ID Classification Loss (BNNeck / Bag of Tricks)
    - PK-Sampler: Mỗi batch lấy P người, mỗi người K ảnh
    - Xuất trọng số đã fine-tune vào: models/reid/reid_mobilenetv3.pth
"""

import os
import sys
import glob
import random
from collections import defaultdict
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms as T
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# Thêm project root vào path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.logger import get_logger

logger = get_logger("train_reid")

# Cấu hình huấn luyện
DATA_ROOT = "datasets/Market-1501-v15.09.15/bounding_box_train"
SAVE_DIR = "models/reid"
MODEL_NAME = "reid_mobilenetv3.pth"
NUM_EPOCHS = 10
P_PERSONS = 4       # Số người trong 1 batch
K_IMAGES = 4        # Số ảnh mỗi người trong 1 batch -> Batch Size = P * K = 16
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 5e-4
TRIPLET_MARGIN = 0.3
SEED = 42

TRAIN_TRANSFORM = T.Compose([
    T.Resize((256, 128)),
    T.RandomHorizontalFlip(p=0.5),
    T.Pad(10),
    T.RandomCrop((256, 128)),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.RandomErasing(p=0.5, scale=(0.02, 0.2)),
])


class Market1501Dataset(Dataset):
    """
    Dataset loader cho Market-1501 format:
    Tên file dạng: [person_id]_[camera_id]_[sequence]_[frame].jpg
    Ví dụ: 0001_c1s1_001051_00.jpg -> person_id = 1
    """

    def __init__(self, data_root: str, transform=None, dummy_mode: bool = False):
        self.data_root = data_root
        self.transform = transform or TRAIN_TRANSFORM
        self.samples = []  # list of (img_path, label_idx)
        self.pid_to_indices = defaultdict(list)
        self.num_classes = 0

        if dummy_mode or not os.path.exists(data_root):
            logger.info("Market-1501 không tìm thấy tại thư mục gốc, kích hoạt chế độ Demo/Dummy (10 PIDs x 6 ảnh).")
            self._init_dummy(num_pids=10, imgs_per_pid=6)
            return

        self._load_market1501()

    def _init_dummy(self, num_pids: int = 10, imgs_per_pid: int = 6):
        self.num_classes = num_pids
        for pid in range(num_pids):
            for i in range(imgs_per_pid):
                idx = len(self.samples)
                self.samples.append((f"dummy_{pid}_{i}.jpg", pid))
                self.pid_to_indices[pid].append(idx)

    def _load_market1501(self):
        img_paths = glob.glob(os.path.join(self.data_root, "*.jpg"))
        pid_set = set()
        raw_samples = []

        for p in img_paths:
            fname = os.path.basename(p)
            parts = fname.split("_")
            if not parts[0].isdigit():
                continue
            pid = int(parts[0])
            if pid == -1 or pid == 0:
                continue  # Bỏ qua background hoặc junk
            pid_set.add(pid)
            raw_samples.append((p, pid))

        # Map person_id -> class index liên tục từ 0 .. N-1
        pid_map = {pid: idx for idx, pid in enumerate(sorted(pid_set))}
        self.num_classes = len(pid_map)

        for p, pid in raw_samples:
            label_idx = pid_map[pid]
            idx = len(self.samples)
            self.samples.append((p, label_idx))
            self.pid_to_indices[label_idx].append(idx)

        logger.info(f"Đã nạp {len(self.samples)} ảnh Market-1501 với {self.num_classes} Person IDs khác nhau.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        if os.path.exists(path):
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                img = Image.fromarray(np.zeros((256, 128, 3), dtype=np.uint8))
        else:
            # Tạo ảnh giả lập nếu file ảo
            img = Image.fromarray(np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8))

        if self.transform:
            img = self.transform(img)

        return img, label


class PKSampler(Sampler):
    """
    PK Sampler: Chọn P person ngẫu nhiên, mỗi person chọn K ảnh.
    Đảm bảo mỗi batch luôn có đủ cặp positive và negative cho Triplet Loss.
    """

    def __init__(self, dataset: Market1501Dataset, p: int = 4, k: int = 4):
        self.dataset = dataset
        self.p = min(p, dataset.num_classes)
        self.k = k
        self.pids = list(dataset.pid_to_indices.keys())

    def __iter__(self):
        # Lập batch danh sách index
        pids = list(self.pids)
        random.shuffle(pids)

        for i in range(0, len(pids) - self.p + 1, self.p):
            batch_pids = pids[i: i + self.p]
            batch_indices = []
            for pid in batch_pids:
                indices = self.dataset.pid_to_indices[pid]
                if len(indices) >= self.k:
                    sampled = random.sample(indices, self.k)
                else:
                    sampled = random.choices(indices, k=self.k)
                batch_indices.extend(sampled)
            yield from batch_indices

    def __len__(self):
        return (len(self.pids) // self.p) * (self.p * self.k)


class TripletLoss(nn.Module):
    """Online Hard Triplet Mining Loss (Hermans et al., 2017)."""

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Khoảng cách Euclidean giữa các cặp embedding: dist(i, j) = ||e_i - e_j||
        dist = torch.cdist(embeddings, embeddings, p=2)

        # Tạo mask positive/negative
        labels = labels.unsqueeze(0)
        is_pos = (labels == labels.T).float()
        is_neg = (labels != labels.T).float()

        # Hard positive: mẫu cùng ID có khoảng cách xa nhất
        dist_ap = (dist * is_pos).max(dim=1)[0]
        # Hard negative: mẫu khác ID có khoảng cách gần nhất
        dist_an = (dist + is_pos * 1e6).min(dim=1)[0]

        y = torch.ones_like(dist_ap)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


class ReIDModel(nn.Module):
    """Mô hình Re-ID MobileNetV3 với Projection Head và BNNeck."""

    def __init__(self, feature_dim: int = 512, num_classes: int = 10):
        super().__init__()
        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(576, feature_dim),
            nn.BatchNorm1d(feature_dim)
        )
        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)

    def forward(self, x):
        feat_map = self.features(x)
        pooled = self.pool(feat_map).flatten(1)
        emb = self.proj(pooled)
        # L2 normalize
        norms = torch.norm(emb, p=2, dim=1, keepdim=True).clamp(min=1e-6)
        norm_emb = emb / norms
        logits = self.classifier(norm_emb)
        return norm_emb, logits


def train_reid():
    # Cố định Seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Bắt đầu huấn luyện Re-ID trên thiết bị: {device}")

    # Dataset & Sampler
    dataset = Market1501Dataset(DATA_ROOT, dummy_mode=not os.path.exists(DATA_ROOT))
    sampler = PKSampler(dataset, p=P_PERSONS, k=K_IMAGES)
    dataloader = DataLoader(dataset, batch_size=P_PERSONS * K_IMAGES, sampler=sampler, num_workers=0)

    # Model, Loss, Optimizer
    model = ReIDModel(feature_dim=512, num_classes=max(10, dataset.num_classes)).to(device)
    triplet_criterion = TripletLoss(margin=TRIPLET_MARGIN)
    id_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, MODEL_NAME)

    logger.info(f"Bắt đầu huấn luyện Re-ID trong {NUM_EPOCHS} epochs (Batch Size={P_PERSONS*K_IMAGES})...")
    best_loss = 999.0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        batch_count = 0

        for imgs, pids in dataloader:
            imgs = imgs.to(device)
            pids = pids.to(device)

            optimizer.zero_grad()
            embs, logits = model(imgs)

            t_loss = triplet_criterion(embs, pids)
            id_loss = id_criterion(logits, pids)
            loss = t_loss + id_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        scheduler.step()
        avg_loss = total_loss / max(1, batch_count)
        logger.info(f"Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] - Re-ID Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            # Lưu state_dict tương thích với ReIDEmbedder (features + proj)
            checkpoint = {
                "model_state_dict": {
                    **{f"features.{k}": v for k, v in model.features.state_dict().items()},
                    **{f"proj.{k}": v for k, v in model.proj.state_dict().items()}
                },
                "feature_dim": 512,
                "epoch": epoch,
                "loss": best_loss
            }
            torch.save(checkpoint, save_path)
            logger.info(f"⭐ Đã lưu trọng số Re-ID tốt nhất vào: {save_path}")

    logger.info(f"✅ Hoàn tất huấn luyện Re-ID! Checkpoint: {save_path}")
    return save_path


if __name__ == "__main__":
    train_reid()
