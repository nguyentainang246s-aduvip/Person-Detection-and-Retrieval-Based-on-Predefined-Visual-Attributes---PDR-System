"""
training/train_color_head.py
============================
Huấn luyện mạng nơ-ron nhận dạng màu sắc quần áo bằng Deep Learning (Giai đoạn 2).

Các kỹ thuật nâng cao:
  - Data Augmentation chống ngược sáng & bóng đổ:
      + ColorJitter (brightness=0.5, contrast=0.4, saturation=0.3)
      + Synthetic Shadow (bóng gradient giả lập ánh sáng gắt)
      + RandomErasing (mô phỏng che khuất một phần quần áo)
  - Loss: Đa nhiệm (Multi-task CrossEntropyLoss cho Upper & Lower)
"""

import os
import sys
import argparse
import csv
import cv2
import torch

# Fix Windows console UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np

# Thêm thư mục gốc vào PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.attributes.learned_color_head import DualColorNet, COLOR_CLASSES, add_synthetic_shadow
from src.utils.video_utils import open_video, read_frame, crop_person


class LabeledCropDataset(Dataset):
    """Dataset nạp ảnh crop từ Ground Truth CSV và video."""

    def __init__(self, samples: list, transform=None):
        self.samples = samples  # list of (crop_img, upper_label_idx, lower_label_idx)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        crop, u_label, l_label = self.samples[idx]
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        if self.transform:
            img_tensor = self.transform(pil_img)
        else:
            img_tensor = T.ToTensor()(pil_img)
        return img_tensor, torch.tensor(u_label, dtype=torch.long), torch.tensor(l_label, dtype=torch.long)


# Định nghĩa tập track_id độc lập cho Test Set (không dùng để train/val)
TEST_TRACKS_4750042 = {2, 7, 8, 16, 28, 46, 74, 88}
TEST_TRACKS_4750061 = {66, 71, 87, 90, 92, 100, 118, 126}


def load_dataset_from_gt(video_path: str, gt_csv_path: str, max_samples: int = 1200, exclude_tracks: set = None):
    """Trích xuất ảnh crop từ video theo nhãn Ground Truth (loại trừ tập test độc lập)."""
    color_map = {c: i for i, c in enumerate(COLOR_CLASSES)}
    exclude_tracks = exclude_tracks or set()
    gt_rows = []
    with open(gt_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if int(r["person_id"]) not in exclude_tracks:
                gt_rows.append(r)

    # Gom nhóm theo frame_idx để đọc video 1 lần
    by_frame = {}
    for r in gt_rows:
        f_idx = int(r["frame_idx"])
        by_frame.setdefault(f_idx, []).append(r)

    cap, info = open_video(video_path)
    collected = []
    frame_idx = 0

    while True:
        ret, frame = read_frame(cap)
        if not ret:
            break

        if frame_idx in by_frame:
            for r in by_frame[frame_idx]:
                x = int(float(r["x"]))
                y = int(float(r["y"]))
                w = int(float(r["w"]))
                h = int(float(r["h"]))
                crop = crop_person(frame, [x, y, x + w, y + h])
                if crop is not None and crop.shape[0] >= 30 and crop.shape[1] >= 15:
                    u_col = r.get("upper_color", "Other").strip().capitalize()
                    l_col = r.get("lower_color", "Other").strip().capitalize()
                    u_idx = color_map.get(u_col, color_map["Other"])
                    l_idx = color_map.get(l_col, color_map["Other"])
                    collected.append((crop, u_idx, l_idx))

                    if len(collected) >= max_samples:
                        break
        frame_idx += 1
        if len(collected) >= max_samples:
            break

    cap.release()
    return collected


def train_color_head(epochs: int = 8, batch_size: int = 16, lr: float = 1e-3, save_path: str = "models/par/color_head.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Bắt đầu huấn luyện Deep Learning Color Head trên thiết bị: {device}")
    print(f"[*] Loại trừ {len(TEST_TRACKS_4750042) + len(TEST_TRACKS_4750061)} tracks độc lập làm Test Set độc lập.")

    # Thu thập mẫu từ cả 2 video (chỉ lấy train tracks)
    samples = []
    if os.path.exists("data/test_videos/4750042-hd_1920_1080_30fps.mp4") and os.path.exists("data/gt/4750042_gt.csv"):
        print("[*] Đang trích xuất mẫu Train từ Video 1...")
        s1 = load_dataset_from_gt("data/test_videos/4750042-hd_1920_1080_30fps.mp4", "data/gt/4750042_gt.csv", max_samples=400, exclude_tracks=TEST_TRACKS_4750042)
        samples.extend(s1)

    if os.path.exists("data/test_videos/4750061-hd_1920_1080_30fps.mp4") and os.path.exists("data/gt/4750061_gt.csv"):
        print("[*] Đang trích xuất mẫu Train từ Video 2...")
        s2 = load_dataset_from_gt("data/test_videos/4750061-hd_1920_1080_30fps.mp4", "data/gt/4750061_gt.csv", max_samples=400, exclude_tracks=TEST_TRACKS_4750061)
        samples.extend(s2)

    if not samples:
        print("[!] Không tìm thấy mẫu training. Vui lòng kiểm tra file video và GT CSV.")
        return

    print(f"[+] Thu thập tổng cộng {len(samples)} mẫu crop có gán nhãn màu.")

    # Phân chia Train / Val (80% / 20%)
    np.random.seed(42)
    np.random.shuffle(samples)
    split = int(0.8 * len(samples))
    train_samples = samples[:split]
    val_samples = samples[split:]

    # Pipeline Augmentation chống ngược sáng & bóng đổ
    train_transform = T.Compose([
        T.Resize((224, 112)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.5, contrast=0.4, saturation=0.3),
        T.RandomApply([T.Lambda(add_synthetic_shadow)], p=0.35),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.3, scale=(0.02, 0.15))
    ])

    val_transform = T.Compose([
        T.Resize((224, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(LabeledCropDataset(train_samples, train_transform), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(LabeledCropDataset(val_samples, val_transform), batch_size=batch_size, shuffle=False)

    model = DualColorNet(n_classes=len(COLOR_CLASSES), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, u_targets, l_targets in train_loader:
            imgs, u_targets, l_targets = imgs.to(device), u_targets.to(device), l_targets.to(device)
            optimizer.zero_grad()
            u_logits, l_logits = model(imgs)
            loss = criterion(u_logits, u_targets) + criterion(l_logits, l_targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Đánh giá Val
        model.eval()
        u_correct, l_correct, total = 0, 0, 0
        with torch.no_grad():
            for imgs, u_targets, l_targets in val_loader:
                imgs, u_targets, l_targets = imgs.to(device), u_targets.to(device), l_targets.to(device)
                u_logits, l_logits = model(imgs)
                u_preds = u_logits.argmax(dim=1)
                l_preds = l_logits.argmax(dim=1)
                u_correct += (u_preds == u_targets).sum().item()
                l_correct += (l_preds == l_targets).sum().item()
                total += len(u_targets)

        val_acc = ((u_correct + l_correct) / (2.0 * total)) * 100.0 if total > 0 else 0.0
        print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Color Acc: {val_acc:.2f}% (Upper: {u_correct/total*100:.1f}%, Lower: {l_correct/total*100:.1f}%)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    print(f"\n[+] Huấn luyện thành công! Trọng số tối ưu đã lưu tại: {save_path} (Best Val Acc: {best_val_acc:.2f}%)")


if __name__ == "__main__":
    train_color_head(epochs=5, batch_size=16)
