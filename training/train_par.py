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


class PARModel(nn.Module):
    """
    PAR Model = ResNet50 Backbone + FC Classification Head.

    INPUT:  (batch, 3, 224, 112) - person crop đã normalize
    OUTPUT: (batch, 4) - sigmoid probabilities cho 4 attributes
    """

    def __init__(self, n_attrs=4, pretrained=True):
        super().__init__()

        # Load ResNet50 pretrained ImageNet
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Bỏ FC layer cuối của ResNet50 (2048 → 1000)
        # Giữ lại từ conv1 đến avgpool
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        # Output shape: (batch, 2048, 1, 1)

        # FC head mới cho attribute recognition
        self.classifier = nn.Sequential(
            nn.Flatten(),               # (batch, 2048)
            nn.Dropout(p=0.5),          # Regularization
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, n_attrs),
            nn.Sigmoid()               # Output: [0, 1] probability
        )

    def forward(self, x):
        """
        x: (batch, 3, 224, 112)
        return: (batch, 4) sigmoid probs
        """
        features = self.backbone(x)     # (batch, 2048, 1, 1)
        output = self.classifier(features)  # (batch, 4)
        return output


def compute_mean_accuracy(preds, labels, threshold=0.5):
    """
    Tính Mean Accuracy (mA) — metric chuẩn trong PAR.

    mA = (1/N_attrs) × Σ [ (TP_i + TN_i) / (P_i + N_i) ]

    Nghĩa là: accuracy trung bình qua từng attribute.

    INPUT:
        preds:  (N, 4) tensor float32, giá trị 0–1
        labels: (N, 4) tensor float32, giá trị 0 hoặc 1
        threshold: ngưỡng để quyết định 0/1

    OUTPUT:
        ma: float - mean accuracy (0–1)
        per_attr: list[float] - accuracy từng attribute
    """
    pred_binary = (preds >= threshold).float()   # (N, 4)

    per_attr = []
    for i in range(preds.shape[1]):
        correct = (pred_binary[:, i] == labels[:, i]).float().sum()
        acc = correct / len(labels)
        per_attr.append(acc.item())

    ma = sum(per_attr) / len(per_attr)
    return ma, per_attr


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    """Chạy một epoch training."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)         # (batch, 4)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.append(outputs.detach().cpu())
        all_labels.append(labels.cpu())

        # In tiến độ
        if (batch_idx + 1) % 100 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} | "
                  f"Loss: {loss.item():.4f}")

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    ma, per_attr = compute_mean_accuracy(all_preds, all_labels)
    avg_loss = total_loss / len(loader)

    return avg_loss, ma, per_attr


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model trên val/test set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        all_preds.append(outputs.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    ma, per_attr = compute_mean_accuracy(all_preds, all_labels)
    avg_loss = total_loss / len(loader)

    return avg_loss, ma, per_attr


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
    model = PARModel(n_attrs=N_ATTRS, pretrained=True).to(device)

    # Đếm số parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params    : {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # ── Loss & Optimizer ──
    # BCELoss vì đây là multi-label binary classification
    criterion = nn.BCELoss()

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

        # Lưu model tốt nhất
        if val_ma > best_val_ma:
            best_val_ma = val_ma
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_ma": val_ma,
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
