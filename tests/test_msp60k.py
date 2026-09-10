"""
tests/test_msp60k.py
====================
Kiểm thử đơn vị cho Module DataLoader MSP60K (AAAI 2025 / OpenPAR-main).
Kiểm tra tính đúng đắn của shape ảnh, nhãn 4 thuộc tính (female, hat, glasses, backpack),
và các pipeline chuyển đổi.
"""

import os
import pytest
import torch
from training.dataset_msp60k import MSP60KDataset, get_msp60k_dataloaders, MSP60K_TARGET_ATTRS, TARGET_ATTR_NAMES

def test_msp60k_dummy_mode_dataset():
    """Kiểm tra MSP60KDataset ở chế độ dummy (không cần ảnh thật)."""
    dataset = MSP60KDataset(
        pkl_path=None,
        img_dir=None,
        split='train',
        target_only=True,
        dummy_mode=True,
        dummy_size=40
    )
    
    assert len(dataset) == 40
    img, labels = dataset[0]
    
    # Kiểm tra tensor hình ảnh (3, 224, 112)
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 112)
    
    # Kiểm tra nhãn mục tiêu (4 thuộc tính)
    assert isinstance(labels, torch.Tensor)
    assert labels.shape == (4,)
    assert labels.dtype == torch.float32
    for val in labels:
        assert val.item() in (0.0, 1.0)

def test_msp60k_all_57_attributes():
    """Kiểm tra nạp toàn bộ 57 thuộc tính của MSP60K gốc."""
    dataset = MSP60KDataset(
        pkl_path=None,
        img_dir=None,
        split='val',
        target_only=False,
        dummy_mode=True,
        dummy_size=20
    )
    
    assert len(dataset) == 20
    img, labels = dataset[5]
    assert img.shape == (3, 224, 112)
    assert labels.shape == (57,)

def test_msp60k_dataloaders():
    """Kiểm tra batching từ hàm get_msp60k_dataloaders."""
    train_loader, val_loader, test_loader = get_msp60k_dataloaders(
        pkl_path=None,
        img_dir=None,
        batch_size=8,
        target_only=True,
        dummy_mode=True
    )
    
    for loader in [train_loader, val_loader, test_loader]:
        batch_imgs, batch_labels = next(iter(loader))
        assert batch_imgs.shape == (8, 3, 224, 112)
        assert batch_labels.shape == (8, 4)

def test_msp60k_sample_pkl_loading():
    """Kiểm tra nạp từ file sample PKL được sinh bởi convert_msp60k_to_pa100k.py."""
    sample_pkl = "data/sample_dataset_ms_split1.pkl"
    if not os.path.exists(sample_pkl):
        pytest.skip("Chưa tạo file sample_dataset_ms_split1.pkl")
        
    dataset = MSP60KDataset(
        pkl_path=sample_pkl,
        img_dir="data/dummy_imgs",
        split='train',
        target_only=True,
        dummy_mode=False
    )
    
    assert len(dataset) == 350
    img, label = dataset[0]
    assert img.shape == (3, 224, 112)
    assert label.shape == (4,)
