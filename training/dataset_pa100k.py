"""
training/dataset_pa100k.py
============================
DataLoader cho dataset PA-100K dùng để train PAR model.

PA-100K FORMAT:
    datasets/PA100K/
    ├── data/                   ← 100,000 ảnh .jpg
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   └── ...
    └── annotation/
        └── annotation.mat      ← Labels dạng MATLAB .mat file

ANNOTATION.MAT STRUCTURE:
    annotation['train_images_name']  → list tên file train (80,000)
    annotation['val_images_name']    → list tên file val (10,000)
    annotation['test_images_name']   → list tên file test (10,000)
    annotation['train_label']        → numpy (80000, 26) binary labels
    annotation['val_label']          → numpy (10000, 26) binary labels
    annotation['test_label']         → numpy (10000, 26) binary labels

PA-100K 26 ATTRIBUTES (index → tên):
    0: Female          7: Backpack      14: LowerStripe   21: Shorts
    1: AgeOver60       8: HandBag       15: LowerPattern  22: Skirt&Dress
    2: Age18-60        9: ShoulderBag   16: LongCoat      23: boots
    3: AgeLess18      10: Hat           17: Trousers      24: LongHair
    4: Front          11: Glasses       18: Shorts        25: BlackHair
    5: Back           12: HoldObjectsInFront
    6: Side           13: ShortSleeve

ATTRIBUTES CHÚNG TA DÙNG:
    0  → Female (nếu =0 thì Male, nếu =1 thì Female)
    10 → Hat
    11 → Glasses
    7  → Backpack

CÁCH CHẠY (Google Colab):
    from training.dataset_pa100k import PA100KDataset, get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root="/content/drive/MyDrive/PA100K",
        batch_size=32
    )
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

try:
    import scipy.io as sio
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[WARNING] scipy chưa cài. Chạy: pip install scipy")


# ── Mapping index PA-100K sang tên attribute của chúng ta ──
# Chỉ dùng 4 attributes từ PA-100K
PA100K_ATTR_IDX = {
    "female": 0,      # 0=Male, 1=Female
    "hat": 10,        # 0=No, 1=Yes
    "glasses": 11,    # 0=No, 1=Yes
    "backpack": 7,    # 0=No, 1=Yes
}

# Transform cho training (có augmentation)
TRAIN_TRANSFORM = T.Compose([
    T.Resize((256, 128)),           # Resize về 256×128 (portrait)
    T.RandomHorizontalFlip(p=0.5),  # Lật ngang ngẫu nhiên
    T.ColorJitter(                  # Thay đổi màu sắc ngẫu nhiên
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    T.RandomCrop((224, 112)),        # Crop ngẫu nhiên → 224×112
    T.ToTensor(),                    # [H,W,C] uint8 → [C,H,W] float32 [0,1]
    T.Normalize(                     # Chuẩn hóa theo ImageNet stats
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Transform cho val/test (không augmentation)
EVAL_TRANSFORM = T.Compose([
    T.Resize((224, 112)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


class PA100KDataset(Dataset):
    """
    PyTorch Dataset cho PA-100K.

    LUỒNG XỬ LÝ:
        annotation.mat
             ↓
        Đọc danh sách ảnh + labels
             ↓
        __getitem__(idx):
            Đọc ảnh từ disk
             ↓
            Apply transform
             ↓
            Lấy 4 label {female, hat, glasses, backpack}
             ↓
        Trả về (image_tensor, label_tensor)

    VÍ DỤ:
        dataset = PA100KDataset(data_root="datasets/PA100K", split="train")
        img, labels = dataset[0]
        print(img.shape)    # torch.Size([3, 224, 112])
        print(labels)       # tensor([0., 1., 0., 1.]) = Male, Hat=Yes, Glasses=No, Backpack=Yes
    """

    def __init__(self, data_root: str, split: str = "train", transform=None):
        """
        INPUT:
            data_root: Đường dẫn đến thư mục PA100K/
                       Phải có: data/ và annotation/annotation.mat
            split: "train", "val", hoặc "test"
            transform: torchvision transform (None → dùng mặc định)
        """
        assert split in ("train", "val", "test"), f"split phải là train/val/test, got: {split}"
        assert SCIPY_AVAILABLE, "Cần cài scipy: pip install scipy"

        self.data_root = data_root
        self.split = split
        self.img_dir = os.path.join(data_root, "data")

        # Transform
        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = TRAIN_TRANSFORM
        else:
            self.transform = EVAL_TRANSFORM

        # Đọc annotation .mat
        ann_path = os.path.join(data_root, "annotation", "annotation.mat")
        if not os.path.exists(ann_path):
            raise FileNotFoundError(
                f"Không tìm thấy annotation.mat tại: {ann_path}\n"
                f"Hãy đảm bảo dataset PA-100K đã được giải nén đúng cách."
            )

        ann = sio.loadmat(ann_path)

        # Đọc danh sách tên file và labels theo split
        split_map = {
            "train": ("train_images_name", "train_label"),
            "val":   ("val_images_name",   "val_label"),
            "test":  ("test_images_name",  "test_label"),
        }
        name_key, label_key = split_map[split]

        # Tên file: dạng array of array, cần flatten
        raw_names = ann[name_key]
        self.image_names = [str(raw_names[i][0][0]) for i in range(len(raw_names))]

        # Labels: numpy array (N, 26)
        all_labels = ann[label_key].astype(np.float32)  # (N, 26)

        # Chỉ lấy 4 attributes cần dùng
        attr_indices = list(PA100K_ATTR_IDX.values())  # [0, 10, 11, 7]
        self.labels = all_labels[:, attr_indices]       # (N, 4)

        print(f"[PA100K] Split={split}: {len(self.image_names)} ảnh | "
              f"Labels shape: {self.labels.shape}")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        """
        OUTPUT:
            image: torch.Tensor (3, 224, 112) float32, normalized
            label: torch.Tensor (4,) float32
                   [female, hat, glasses, backpack]
                   Giá trị 0.0 hoặc 1.0
        """
        # Đọc ảnh
        img_path = os.path.join(self.img_dir, self.image_names[idx])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback: ảnh đen nếu không đọc được
            print(f"[WARNING] Không đọc được ảnh: {img_path} | {e}")
            image = Image.new("RGB", (128, 256), color=(0, 0, 0))

        # Apply transform
        image = self.transform(image)

        # Label
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label


def get_dataloaders(data_root: str, batch_size: int = 32, num_workers: int = 2):
    """
    Tạo DataLoader cho train, val, test.

    INPUT:
        data_root: Đường dẫn thư mục PA100K/
        batch_size: Số ảnh mỗi batch (32 cho Colab T4, 16 cho GTX 1650)
        num_workers: Số worker đọc dữ liệu song song
                     (0 trên Windows nếu có lỗi multiprocessing)

    OUTPUT:
        train_loader, val_loader, test_loader

    VÍ DỤ (Google Colab):
        train_loader, val_loader, test_loader = get_dataloaders(
            data_root="/content/drive/MyDrive/PA100K",
            batch_size=32,
            num_workers=2
        )
        for images, labels in train_loader:
            # images: (32, 3, 224, 112)
            # labels: (32, 4)
            break
    """
    train_ds = PA100KDataset(data_root, split="train")
    val_ds   = PA100KDataset(data_root, split="val")
    test_ds  = PA100KDataset(data_root, split="test")

    # Windows cần pin_memory=False và num_workers=0 nếu gặp lỗi
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    print(f"\n[DataLoaders Ready]")
    print(f"  Train: {len(train_ds)} ảnh, {len(train_loader)} batches")
    print(f"  Val  : {len(val_ds)} ảnh, {len(val_loader)} batches")
    print(f"  Test : {len(test_ds)} ảnh, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


def explore_dataset(data_root: str):
    """
    Phân tích nhanh dataset PA-100K.
    Chạy để hiểu distribution của các attribute.

    VÍ DỤ:
        explore_dataset("datasets/PA100K")
    """
    print("=" * 50)
    print("  PA-100K Dataset Exploration")
    print("=" * 50)

    train_ds = PA100KDataset(data_root, split="train")

    labels = train_ds.labels  # (80000, 4)
    attr_names = list(PA100K_ATTR_IDX.keys())

    print(f"\nTổng số ảnh train: {len(train_ds)}")
    print(f"\nPhân phối từng attribute:")
    print(f"{'Attribute':<15} {'Positive':>10} {'Negative':>10} {'Ratio':>10}")
    print("-" * 50)

    for i, name in enumerate(attr_names):
        pos = int(labels[:, i].sum())
        neg = len(labels) - pos
        ratio = pos / len(labels) * 100
        print(f"{name:<15} {pos:>10,} {neg:>10,} {ratio:>9.1f}%")
