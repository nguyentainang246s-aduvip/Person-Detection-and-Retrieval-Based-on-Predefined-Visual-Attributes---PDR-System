"""
tests/test_track_memory.py
==========================
Unit tests cho TrackMemoryManager và GlobalReIDGallery.
"""
import pytest
import numpy as np
from src.retrieval.track_memory import TrackMemoryManager
from src.retrieval.global_gallery import GlobalReIDGallery


class TestTrackMemoryManager:
    def test_basic_crud(self):
        mem = TrackMemoryManager()
        assert len(mem) == 0

        mem.update(1, {"last_updated": 10, "gender": "Male", "crop": np.zeros((10, 10, 3))})
        assert len(mem) == 1
        assert mem.get(1)["gender"] == "Male"

    def test_stale_eviction_to_ghost(self):
        mem = TrackMemoryManager()
        mem.update(1, {"last_updated": 10, "crop": np.zeros((10, 10, 3))})
        # frame 400: age is 390 > 300 TTL
        mem.evict_stale(frame_idx=400)

        assert mem.get(1) is None
        assert 1 in mem.ghosts
        assert mem.ghosts[1]["crop"] is None  # RAM freed

    def test_ghost_window(self):
        mem = TrackMemoryManager()
        mem.update(1, {"last_updated": 100, "crop": None})
        mem.evict_stale(frame_idx=450)  # evicted to ghost, last_updated is 100

        # at frame 200: 200 - 100 = 100 > 60
        valid = mem.get_valid_ghosts(frame_idx=200)
        assert 1 not in valid

    def test_clear(self):
        mem = TrackMemoryManager()
        mem.update(1, {"last_updated": 10})
        mem.clear()
        assert len(mem) == 0
        assert len(mem.ghosts) == 0


class TestGlobalReIDGallery:
    def test_query_or_register_new(self):
        gallery = GlobalReIDGallery(similarity_threshold=0.7)
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)

        gid, is_new = gallery.query_or_register("cam1", local_track_id=1, embedding=emb)
        assert gid == 1
        assert is_new is True

        # Same camera and track returns same gid
        gid_same, is_new_same = gallery.query_or_register("cam1", local_track_id=1, embedding=emb)
        assert gid_same == 1
        assert is_new_same is False

    def test_cross_camera_match(self):
        gallery = GlobalReIDGallery(similarity_threshold=0.7)
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)

        # cam1 registers person
        gid1, is_new1 = gallery.query_or_register("cam1", local_track_id=10, embedding=emb)
        assert is_new1 is True

        # cam2 observes same person (very close embedding)
        emb_cam2 = emb + np.random.normal(0, 0.01, size=512).astype(np.float32)
        emb_cam2 /= np.linalg.norm(emb_cam2)

        gid2, is_new2 = gallery.query_or_register("cam2", local_track_id=5, embedding=emb_cam2)
        assert gid2 == gid1  # Re-ID match across cameras!
        assert is_new2 is False
