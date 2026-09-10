"""
tests/test_upar.py
==================
Unit tests cho UPAR dataset loader và AttributeRecognizer với UPAR 40-attr format.
"""
import pytest
import os
import torch
import numpy as np
from training.dataset_upar import UPARDataset, get_upar_dataloaders
from src.attributes.par_model import AttributeRecognizer, build_par_model


class TestUPARDataset:
    def test_dummy_dataset_shapes(self):
        ds = UPARDataset(data_root="dummy_path", split="train", only_pdr_attrs=True, dummy_mode=True)
        assert len(ds) == 100
        img, target = ds[0]
        assert img.shape == (3, 224, 112)
        assert target.shape == (4,)

    def test_dummy_dataset_40_attrs(self):
        ds = UPARDataset(data_root="dummy_path", split="val", only_pdr_attrs=False, dummy_mode=True)
        img, target = ds[0]
        assert target.shape == (40,)

    def test_dataloaders(self):
        train_l, val_l, test_l = get_upar_dataloaders("dummy_path", batch_size=8, dummy_mode=True)
        batch_imgs, batch_targets = next(iter(train_l))
        assert batch_imgs.shape[0] == 8
        assert batch_targets.shape == (8, 4)


class TestUPARModelCompatibility:
    @pytest.fixture
    def mock_upar_checkpoint(self, tmp_path):
        """Tạo checkpoint giả lập 40 thuộc tính chuẩn UPAR."""
        model_40 = build_par_model(backbone="resnet50", n_attrs=40, pretrained=False)
        ckpt_path = os.path.join(tmp_path, "mock_upar_40.pth")
        torch.save({
            "model_state_dict": model_40.state_dict(),
            "n_attrs": 40,
            "val_ma": 0.915
        }, ckpt_path)
        return ckpt_path

    def test_attribute_recognizer_auto_detect_upar(self, mock_upar_checkpoint):
        recognizer = AttributeRecognizer(weights_path=mock_upar_checkpoint, backbone="resnet50")
        assert recognizer.n_attrs == 40
        assert recognizer.attr_indices["female"] == 0
        assert recognizer.attr_indices["backpack"] == 35
        assert recognizer.attr_indices["glasses"] == 37
        assert recognizer.attr_indices["hat"] == 38

        # Kiểm tra predict đơn crop
        dummy_crop = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
        pred = recognizer.predict(dummy_crop)

        assert "gender" in pred
        assert pred["gender"] in ["Male", "Female"]
        assert "hat" in pred and isinstance(pred["hat"], bool)
        assert "glasses" in pred and isinstance(pred["glasses"], bool)
        assert "backpack" in pred and isinstance(pred["backpack"], bool)

        # Kiểm tra predict_batch
        batch_preds = recognizer.predict_batch([dummy_crop, dummy_crop])
        assert len(batch_preds) == 2
        assert "gender" in batch_preds[0]
        assert "backpack" in batch_preds[1]
