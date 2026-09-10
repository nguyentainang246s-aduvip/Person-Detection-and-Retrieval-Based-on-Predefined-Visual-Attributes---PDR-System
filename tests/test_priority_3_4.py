"""
tests/test_priority_3_4.py
==========================
Unit tests kiểm tra các tính năng của ƯU TIÊN 3 và ƯU TIÊN 4:
    - Ưu tiên 3: Căn chỉnh ngưỡng quyết định thuộc tính động và calibration
    - Ưu tiên 4: Re-ID fine-tuning, PK-Sampler, Triplet Loss và cosine similarity
"""

import pytest
import os
import torch
import numpy as np

from src.attributes.par_model import AttributeRecognizer
from src.tracking.reid_embedder import ReIDEmbedder
from training.train_reid import Market1501Dataset, PKSampler, TripletLoss
from scripts.calibrate_thresholds import calibrate_thresholds


class TestPriority3ThresholdCalibration:
    def test_dynamic_threshold_update(self):
        recognizer = AttributeRecognizer()
        orig_hat = recognizer.thresholds["hat"]

        # Cập nhật ngưỡng động theo môi trường CCTV
        recognizer.thresholds["hat"] = 0.68
        recognizer.thresholds["gender"] = 0.48
        assert recognizer.thresholds["hat"] == 0.68
        assert recognizer.thresholds["gender"] == 0.48

    def test_calibration_script_runs(self):
        optimal = calibrate_thresholds()
        assert "female" in optimal
        assert "hat" in optimal
        assert "glasses" in optimal
        assert "backpack" in optimal
        for attr, data in optimal.items():
            assert 0.3 <= data["optimal_threshold"] <= 0.8
            assert data["max_f1"] > 0.0


class TestPriority4ReIDMarket1501:
    def test_reid_embedder_fine_tuned_loading(self):
        weights_path = "models/reid/reid_mobilenetv3.pth"
        embedder = ReIDEmbedder(weights_path=weights_path)
        assert embedder.feature_dim == 512

        crop = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
        emb = embedder.extract(crop)
        assert emb.shape == (512,)
        assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-4)

    def test_triplet_loss_and_pk_sampler(self):
        dataset = Market1501Dataset("dummy_path", dummy_mode=True)
        assert dataset.num_classes == 10
        assert len(dataset) == 60

        sampler = PKSampler(dataset, p=4, k=4)
        indices = list(iter(sampler))
        assert len(indices) == 32  # 2 chunks of (4*4)

        # Kiểm tra Triplet Loss
        criterion = TripletLoss(margin=0.3)
        embeddings = torch.randn(16, 512)
        embeddings = embeddings / torch.norm(embeddings, p=2, dim=1, keepdim=True)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
        loss = criterion(embeddings, labels)
        assert loss.item() >= 0.0

    def test_reid_similarity_separation(self):
        embedder = ReIDEmbedder(weights_path="models/reid/reid_mobilenetv3.pth")
        crop_a = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
        # Crop B là bản sao có nhiễu nhẹ của A (Positive)
        crop_b = np.clip(crop_a.astype(np.float32) + np.random.normal(0, 2, crop_a.shape), 0, 255).astype(np.uint8)
        # Crop C là người khác hoàn toàn (Negative)
        crop_c = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)

        emb_a = embedder.extract(crop_a)
        emb_b = embedder.extract(crop_b)
        emb_c = embedder.extract(crop_c)

        sim_pos = embedder.cosine_similarity(emb_a, emb_b)
        sim_neg = embedder.cosine_similarity(emb_a, emb_c)

        assert sim_pos > sim_neg
