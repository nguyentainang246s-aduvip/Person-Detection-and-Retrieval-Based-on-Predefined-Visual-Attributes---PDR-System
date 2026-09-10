"""
training/train_upar.py
======================
Script huấn luyện mô hình PAR trên bộ dữ liệu chuẩn hóa quốc tế UPAR Benchmark.
Tích hợp:
    - Focal Loss (chống mất cân bằng dữ liệu)
    - Seed cố định 42 (Reproducible kết quả)
    - StepLR Scheduler
    - Tính toán Mean Accuracy (mA) trên tập Validation
    - Xuất trọng số chuẩn vào models/par/par_resnet50.pth (hoặc par_upar_resnet50.pth)
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

# Thêm thư mục gốc vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.dataset_upar import get_upar_dataloaders, PDR_TARGET_ATTRS
from src.attributes.par_model import build_par_model, FocalLoss
from src.utils.logger import get_logger

logger = get_logger("train_upar")

# ══════════════════════════════════════════════════════════
# CẤU HÌNH HUẤN LUYỆN
# ══════════════════════════════════════════════════════════
DATA_ROOT = "datasets/UPAR"               # Đường dẫn tập dữ liệu UPAR
SAVE_DIR = "models/par"                   # Thư mục lưu checkpoint
MODEL_NAME = "par_upar_resnet50.pth"      # Tên checkpoint xuất ra
BACKBONE = "resnet50"                     # 'resnet50' hoặc 'mobilenetv3'

BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
LOSS_TYPE = "focal"                       # 'focal' hoặc 'bce'
FOCAL_GAMMA = 2.0
ONLY_PDR_ATTRS = True                     # True: train 4 thuộc tính cốt lõi; False: train full 40 thuộc tính


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, attr_names):
    model.eval()
    all_preds, all_targets = [], []

    for images, targets in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(np.float32)

        all_preds.append(preds)
        all_targets.append(targets.numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # Tính Mean Accuracy (mA)
    accs = []
    per_class_acc = {}
    for i, name in enumerate(attr_names):
        pos_mask = all_targets[:, i] == 1
        neg_mask = all_targets[:, i] == 0

        pos_acc = (all_preds[pos_mask, i] == 1).mean() if pos_mask.sum() > 0 else 0.0
        neg_acc = (all_preds[neg_mask, i] == 0).mean() if neg_mask.sum() > 0 else 0.0
        class_ma = 0.5 * (pos_acc + neg_acc)
        accs.append(class_ma)
        per_class_acc[name] = float(class_ma)

    mean_acc = float(np.mean(accs))
    return mean_acc, per_class_acc


def main():
    # Cố định Seed
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Bắt đầu huấn luyện UPAR trên thiết bị: {device}")

    # DataLoaders
    train_loader, val_loader, _ = get_upar_dataloaders(
        data_root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=0 if os.name == "nt" else 2,
        only_pdr_attrs=ONLY_PDR_ATTRS,
        dummy_mode=not os.path.exists(DATA_ROOT)
    )

    attr_names = ["female", "hat", "glasses", "backpack"] if ONLY_PDR_ATTRS else [f"attr_{i}" for i in range(40)]
    n_attrs = len(attr_names)

    # Xây dựng mô hình
    model = build_par_model(backbone=BACKBONE, n_attrs=n_attrs, pretrained=True).to(device)

    # Tính pos_weight cho Focal Loss
    try:
        train_labels = torch.tensor(train_loader.dataset.labels, dtype=torch.float32)
        pos = train_labels.sum(dim=0)
        neg = len(train_labels) - pos
        pos_weight = (neg / pos.clamp(min=1.0)).to(device)
        alpha_t = (1.0 / (1.0 + pos_weight)).to(device)
    except Exception:
        alpha_t = None

    if LOSS_TYPE == "focal":
        criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=alpha_t)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    best_ma = 0.0
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, MODEL_NAME)

    logger.info(f"Huấn luyện trong {NUM_EPOCHS} epochs...")
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_ma, class_acc = evaluate(model, val_loader, device, attr_names)
        scheduler.step()
        elapsed = time.time() - t0

        logger.info(
            f"Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] ({elapsed:.1f}s) | "
            f"Loss: {train_loss:.4f} | Val mA: {val_ma * 100:.2f}%"
        )

        if val_ma > best_ma:
            best_ma = val_ma
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "val_ma": best_ma,
                "n_attrs": n_attrs,
                "attr_names": attr_names,
                "epoch": epoch,
                "thresholds": {"gender": 0.5, "hat": 0.55, "glasses": 0.5, "backpack": 0.5}
            }
            torch.save(checkpoint, save_path)
            logger.info(f"⭐ Lưu checkpoint tốt nhất (mA={best_ma * 100:.2f}%) tại: {save_path}")

    logger.info(f"✅ Hoàn thành huấn luyện UPAR! Best mA: {best_ma * 100:.2f}%")


if __name__ == "__main__":
    main()
