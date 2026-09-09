"""
training/train_par.py
======================
Script train PAR (Person Attribute Recognition) model trên PA-100K.

CHẠY TRÊN GOOGLE COLAB:
    1. Upload file này và dataset_pa100k.py lên Colab
    2. Mount Google Drive
    3. Thay đổi DATA_ROOT và SAVE_PATH bên dưới
    4. Chạy: !python train_par.py

OUTPUT:
    models/par/par_resnet50.pth     ← Model tốt nhất (theo val mA)
    models/par/training_log.csv     ← Loss và metrics theo epoch

KIẾN TRÚC:
    ResNet50 (ImageNet pretrained)
        └── FC Head: 2048 → 512 → 4 outputs
            ├── gender   (Sigmoid → 0/1)
            ├── hat      (Sigmoid → 0/1)
            ├── glasses  (Sigmoid → 0/1)
            └── backpack (Sigmoid → 0/1)

THỜI GIAN TRAINING (ước tính):
    Colab T4 GPU : ~30-45 phút (30 epochs, batch=32)
    GTX 1650     : ~90-120 phút (30 epochs, batch=16)
    CPU only     : KHÔNG KHUYẾN KHÍCH (quá chậm)
"""

import os
import sys
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

# Thêm thư mục gốc vào path để import dataset
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.dataset_pa100k import get_dataloaders

# ══════════════════════════════════════════════════════════
# CẤU HÌNH – THAY ĐỔI Ở ĐÂY
# ══════════════════════════════════════════════════════════

# Đường dẫn dataset PA-100K
# Colab: "/content/drive/MyDrive/PA100K"
# Local: "datasets/PA100K"
DATA_ROOT = "datasets/PA100K"

# Nơi lưu model sau khi train
SAVE_DIR = "models/par"
MODEL_NAME = "par_resnet50.pth"

# Hyperparameters
BATCH_SIZE = 32       # Giảm xuống 16 nếu hết VRAM
NUM_EPOCHS = 30
LR_BACKBONE = 1e-4    # LR thấp cho backbone (đã pretrained)
LR_FC = 1e-3          # LR cao hơn cho FC head (mới)
LR_STEP = 10          # Giảm LR sau mỗi 10 epoch
LR_GAMMA = 0.1        # LR × 0.1 mỗi step
NUM_WORKERS = 2       # 0 nếu gặp lỗi multiprocessing trên Windows

# Số lượng attribute đầu ra
N_ATTRS = 4           # female, hat, glasses, backpack
ATTR_NAMES = ["female", "hat", "glasses", "backpack"]

# ══════════════════════════════════════════════════════════


from src.attributes.par_model import ResNet50PAR


def compute_par_metrics(preds: torch.Tensor, labels: torch.Tensor,
                        threshold: float = 0.5) -> dict:
    """
    Metric chuẩn PAR (Li et al., PA-100K):
      mA  = (1/N) * Σ_attrs 0.5 * (TP/P + TN/N)  (Balanced Mean Accuracy)
      + per-attribute Precision / Recall / F1 (cho dữ liệu mất cân bằng)
    """
    pred_bin = (preds >= threshold).float()
    n_attrs = preds.shape[1]
    per_attr = {}
    ma_sum = 0.0

    for i in range(n_attrs):
        P = labels[:, i].sum().clamp(min=1)
        N = (1 - labels[:, i]).sum().clamp(min=1)
        TP = (pred_bin[:, i] * labels[:, i]).sum()
        TN = ((1 - pred_bin[:, i]) * (1 - labels[:, i])).sum()

        precision = TP / pred_bin[:, i].sum().clamp(min=1)
        recall    = TP / P
        f1        = 2 * precision * recall / (precision + recall).clamp(min=1e-8)

        per_attr[i] = {
            "acc": ((TP + TN) / len(labels)).item(),
            "precision": precision.item(),
            "recall": recall.item(),
            "f1": f1.item(),
        }
        ma_sum += 0.5 * (TP / P + TN / N)

    ma = (ma_sum / n_attrs).item()
    mean_f1 = sum(a["f1"] for a in per_attr.values()) / n_attrs
    return {"mA": ma, "mean_F1": mean_f1, "per_attr": per_attr}


@torch.no_grad()
def tune_thresholds(model, loader, device, attr_names):
    """Chọn threshold tối ưu F1 cho từng attribute trên val set."""
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in loader:
        logits = model(images.to(device)).cpu()
        all_logits.append(logits)
        all_labels.append(labels)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = 1.0 / (1.0 + torch.exp(-logits))

    best_thr = {}
    for i, name in enumerate(attr_names):
        best_f1, best_t = 0.0, 0.5
        for t in np.arange(0.10, 0.90, 0.02):
            p = (probs[:, i] >= t).float()
            tp = (p * labels[:, i]).sum()
            prec = tp / p.sum().clamp(min=1)
            rec  = tp / labels[:, i].sum().clamp(min=1)
            f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-8)
            if f1 > best_f1:
                best_f1, best_t = f1.item(), t
        best_thr[name] = round(float(best_t), 2)
        print(f"  [Tune Threshold] {name:<10}: best thr={best_t:.2f} (F1={best_f1:.4f})")
    return best_thr


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    """Chạy một epoch training."""
    model.train()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass (model trả về logits thô)
        optimizer.zero_grad()
        logits = model(images)         # (batch, 4) logits
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        probs = 1.0 / (1.0 + torch.exp(-logits.detach()))
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

        # In tiến độ
        if (batch_idx + 1) % 100 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} | "
                  f"Loss: {loss.item():.4f}")

    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_par_metrics(all_probs, all_labels)
    avg_loss = total_loss / len(loader)

    return avg_loss, metrics["mA"], [metrics["per_attr"][i]["acc"] for i in range(len(metrics["per_attr"]))]


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model trên val/test set."""
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        probs = 1.0 / (1.0 + torch.exp(-logits))
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_par_metrics(all_probs, all_labels)
    avg_loss = total_loss / len(loader)

    return avg_loss, metrics["mA"], [metrics["per_attr"][i]["acc"] for i in range(len(metrics["per_attr"]))]


def train():
    """Main training function."""

    print("=" * 60)
    print("  PAR Model Training – ResNet50 on PA-100K")
    print("=" * 60)

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory // (1024**3)} GB")

    # ── Data ──
    print(f"\n  Loading dataset from: {DATA_ROOT}")
    train_loader, val_loader, _ = get_dataloaders(
        DATA_ROOT, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    # ── Model ──
    print("\n  Building model...")
    model = ResNet50PAR(n_attrs=N_ATTRS, pretrained=True).to(device)

    # Đếm số parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params    : {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # ── Loss & Optimizer ──
    # Tính pos_weight cho BCEWithLogitsLoss để bù đắp mất cân bằng mẫu (đặc biệt Hat, Glasses, Backpack)
    try:
        train_ds = train_loader.dataset
        labels_all = torch.tensor(train_ds.labels, dtype=torch.float32)
        pos = labels_all.sum(dim=0)
        neg = len(train_ds) - pos
        pos_weight = (neg / pos.clamp(min=1.0)).to(device)
        print(f"  Computed pos_weight: " + " | ".join(
            f"{ATTR_NAMES[i]}={pos_weight[i]:.2f}" for i in range(N_ATTRS)
        ))
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    except Exception as e:
        print(f"  [Warning] Không thể tự động tính pos_weight ({e}), dùng BCEWithLogitsLoss mặc định.")
        criterion = nn.BCEWithLogitsLoss()

    # Dùng 2 learning rate khác nhau:
    # - Backbone: lr thấp (đã pretrained, không cần học nhiều)
    # - FC head: lr cao (cần học nhanh từ đầu)
    optimizer = Adam([
        {"params": model.backbone.parameters(), "lr": LR_BACKBONE},
        {"params": model.classifier.parameters(), "lr": LR_FC},
    ])

    scheduler = StepLR(optimizer, step_size=LR_STEP, gamma=LR_GAMMA)

    # ── Logging ──
    os.makedirs(SAVE_DIR, exist_ok=True)
    log_path = os.path.join(SAVE_DIR, "training_log.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "epoch", "train_loss", "train_mA",
        "val_loss", "val_mA",
        *[f"val_{n}" for n in ATTR_NAMES],
        "lr"
    ])

    # ── Training Loop ──
    best_val_ma = 0.0
    best_model_path = os.path.join(SAVE_DIR, MODEL_NAME)

    print(f"\n  Starting training for {NUM_EPOCHS} epochs...")
    print(f"  Model will be saved to: {best_model_path}\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss, train_ma, train_per = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        # Validate
        val_loss, val_ma, val_per = evaluate(model, val_loader, criterion, device)

        # LR step
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time

        # In kết quả
        print(f"\nEpoch {epoch:3d}/{NUM_EPOCHS} | Time: {elapsed:.0f}s")
        print(f"  Train → Loss: {train_loss:.4f} | mA: {train_ma*100:.2f}%")
        print(f"  Val   → Loss: {val_loss:.4f} | mA: {val_ma*100:.2f}%")
        print(f"  Per-attr val: " + " | ".join(
            f"{ATTR_NAMES[i]}: {val_per[i]*100:.1f}%" for i in range(N_ATTRS)
        ))
        print(f"  LR: {current_lr:.2e}")

        # Lưu model tốt nhất & tune thresholds
        if val_ma > best_val_ma:
            best_val_ma = val_ma
            best_thr = tune_thresholds(model, val_loader, device, ATTR_NAMES)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_ma": val_ma,
                "thresholds": best_thr,
                "attr_names": ATTR_NAMES,
            }, best_model_path)
            print(f"  ✅ Saved best model! (val mA: {val_ma*100:.2f}%)")

        # Ghi log
        log_writer.writerow([
            epoch, train_loss, train_ma,
            val_loss, val_ma, *val_per, current_lr
        ])
        log_file.flush()

    log_file.close()
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best val mA: {best_val_ma*100:.2f}%")
    print(f"  Model saved: {best_model_path}")
    print(f"  Log saved  : {log_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
