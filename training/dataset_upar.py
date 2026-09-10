"""
training/dataset_upar.py
========================
DataLoader cho dataset chuẩn UPAR (Unified Pedestrian Attribute Recognition).
Hỗ trợ đọc dữ liệu chuẩn hóa đa nguồn (PA-100K, Market-1501, PETA, RAPv2).

UPAR BENCHMARK:
    - 40 thuộc tính nhị phân chuẩn hóa (WACV 2023 / 2024 Challenge)
    - Định dạng: CSV hoặc Pickle (.pkl) kèm thư mục ảnh

CÁC THUỘC TÍNH MỤC TIÊU CHO PDR-SYSTEM:
    - Female (Index 0: 'gender_female')
    - Hat (Index 38: 'accessory_hat')
    - Glasses (Index 37: 'accessory_glasses')
    - Backpack (Index 35: 'accessory_backpack')
"""

import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from src.utils.logger import get_logger

logger = get_logger("dataset_upar")

# Danh sách 40 thuộc tính chuẩn của UPAR Benchmark
UPAR_ALL_40_ATTRS = [
    "gender_female",
    "age_young", "age_adult", "age_old",
    "hair_short", "hair_long", "hair_bald",
    "upperbody_color_black", "upperbody_color_blue", "upperbody_color_brown",
    "upperbody_color_green", "upperbody_color_grey", "upperbody_color_orange",
    "upperbody_color_pink", "upperbody_color_purple", "upperbody_color_red",
    "upperbody_color_white", "upperbody_color_yellow", "upperbody_color_other",
    "upperbody_length_short",
    "lowerbody_color_black", "lowerbody_color_blue", "lowerbody_color_brown",
    "lowerbody_color_green", "lowerbody_color_grey", "lowerbody_color_orange",
    "lowerbody_color_pink", "lowerbody_color_purple", "lowerbody_color_red",
    "lowerbody_color_white", "lowerbody_color_yellow", "lowerbody_color_other",
    "lowerbody_length_short", "lowerbody_type_trousers", "lowerbody_type_skirt&dress",
    "accessory_backpack", "accessory_bag", "accessory_glasses", "accessory_hat", "accessory_other"
]

# 4 thuộc tính cốt lõi cho PDR-System
PDR_TARGET_ATTRS = ["gender_female", "accessory_hat", "accessory_glasses", "accessory_backpack"]

# Data Augmentation mạnh mẽ cho training
UPAR_TRAIN_TRANSFORM = T.Compose([
    T.Resize((256, 128)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    T.RandomPerspective(distortion_scale=0.15, p=0.3),
    T.RandomCrop((224, 112)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
])

# Transform cho validation & testing
UPAR_VAL_TRANSFORM = T.Compose([
    T.Resize((256, 128)),
    T.CenterCrop((224, 112)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class UPARDataset(Dataset):
    """
    Dataset loader cho dữ liệu UPAR.
    Tự động tìm kiếm file chú thích (.csv hoặc .pkl) trong thư mục data_root.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transform: T.Compose = None,
        only_pdr_attrs: bool = True,
        dummy_mode: bool = False
    ):
        self.data_root = data_root
        self.split = split
        self.transform = transform or (UPAR_TRAIN_TRANSFORM if split == "train" else UPAR_VAL_TRANSFORM)
        self.only_pdr_attrs = only_pdr_attrs
        self.image_paths = []
        self.labels = np.empty((0, 4 if only_pdr_attrs else 40), dtype=np.float32)

        if dummy_mode or not os.path.exists(data_root):
            logger.info(f"Khởi tạo UPARDataset ở chế độ Dummy/Demo (split={split}).")
            self._init_dummy(num_samples=100)
            return

        self._load_dataset()

    def _init_dummy(self, num_samples: int = 100):
        """Tạo dữ liệu giả lập cho unit test và kiểm tra pipeline."""
        self.image_paths = [f"dummy_{self.split}_{i}.jpg" for i in range(num_samples)]
        n_cols = 4 if self.only_pdr_attrs else 40
        self.labels = (np.random.rand(num_samples, n_cols) > 0.6).astype(np.float32)

    def _load_dataset(self):
        """Đọc file chú thích CSV hoặc PKL của UPAR."""
        # Tìm các file chú thích phổ biến: upar_train.csv, train.csv, annotations.csv...
        candidates = [
            os.path.join(self.data_root, f"upar_{self.split}.csv"),
            os.path.join(self.data_root, f"{self.split}.csv"),
            os.path.join(self.data_root, "annotations", f"{self.split}.csv"),
            os.path.join(self.data_root, f"{self.split}.pkl"),
        ]

        annot_file = None
        for c in candidates:
            if os.path.exists(c):
                annot_file = c
                break

        if annot_file is None:
            csv_matches = glob.glob(os.path.join(self.data_root, f"*{self.split}*.csv"))
            if csv_matches:
                annot_file = csv_matches[0]

        if annot_file is None:
            logger.warning(f"Không tìm thấy file annotation cho '{self.split}' tại {self.data_root}. Dùng dummy mode.")
            self._init_dummy(num_samples=50)
            return

        logger.info(f"Đang nạp chú thích UPAR từ: {annot_file}")
        if annot_file.endswith(".pkl"):
            df = pd.read_pickle(annot_file)
        else:
            df = pd.read_csv(annot_file)

        # Tìm cột đường dẫn ảnh
        path_col = None
        for col in ["image_path", "filepath", "img_name", "image_name", "filename", "path"]:
            if col in df.columns:
                path_col = col
                break
        if path_col is None:
            path_col = df.columns[0]

        # Xác định các cột thuộc tính
        if self.only_pdr_attrs:
            target_cols = []
            for attr in PDR_TARGET_ATTRS:
                # Tìm tên cột tương đương (vd: 'accessory_hat', 'hat', 'accessory:hat')
                matched_col = None
                clean_attr = attr.replace("_", "").replace(":", "").lower()
                for c in df.columns:
                    clean_c = str(c).replace("_", "").replace(":", "").lower()
                    if clean_attr in clean_c or clean_c in clean_attr:
                        matched_col = c
                        break
                if matched_col is not None:
                    target_cols.append(matched_col)
                else:
                    logger.warning(f"Không tìm thấy cột thuộc tính '{attr}' trong CSV, tạo cột nhãn 0.")
                    df[attr] = 0
                    target_cols.append(attr)

            self.labels = df[target_cols].values.astype(np.float32)
        else:
            # Lấy toàn bộ 40 cột hoặc các cột nhãn nhị phân
            feature_cols = [c for c in df.columns if c != path_col and df[c].dtype in [int, float, bool, np.int64, np.float64]]
            self.labels = df[feature_cols].values.astype(np.float32)

        # Xây dựng đường dẫn ảnh tuyệt đối
        raw_paths = df[path_col].tolist()
        self.image_paths = []
        for p in raw_paths:
            if os.path.isabs(p) and os.path.exists(p):
                self.image_paths.append(p)
            else:
                full_p = os.path.join(self.data_root, p)
                if not os.path.exists(full_p):
                    # Thử tìm trong thư mục images/ hoặc data/
                    alt_p = os.path.join(self.data_root, "images", os.path.basename(p))
                    if os.path.exists(alt_p):
                        full_p = alt_p
                self.image_paths.append(full_p)

        logger.info(f"Đã nạp {len(self.image_paths)} mẫu từ UPAR ({self.split}), số thuộc tính: {self.labels.shape[1]}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        target = torch.tensor(self.labels[idx], dtype=torch.float32)

        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                img = Image.fromarray(np.zeros((256, 128, 3), dtype=np.uint8))
        else:
            # Tạo ảnh giả lập nếu file ảnh chưa được tải
            img = Image.fromarray(np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8))

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def get_upar_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 2,
    only_pdr_attrs: bool = True,
    dummy_mode: bool = False
):
    """
    Tạo DataLoaders cho Train, Val, Test của UPAR.
    """
    train_ds = UPARDataset(data_root, split="train", only_pdr_attrs=only_pdr_attrs, dummy_mode=dummy_mode)
    val_ds = UPARDataset(data_root, split="val", only_pdr_attrs=only_pdr_attrs, dummy_mode=dummy_mode)
    test_ds = UPARDataset(data_root, split="test", only_pdr_attrs=only_pdr_attrs, dummy_mode=dummy_mode)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
