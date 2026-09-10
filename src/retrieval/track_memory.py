"""
src/retrieval/track_memory.py
=============================
Quản lý bộ nhớ Track: cache attributes, ghost tracks, eviction policy.
Tách từ pipeline.py nhằm tối ưu hóa kiến trúc theo Single Responsibility Principle.
"""

from typing import Dict, Any, Optional
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("track_memory")

TRACK_MEMORY_TTL_FRAMES = 300
TRACK_MEMORY_MAX_SIZE = 500
GHOST_MAX_SIZE = 50
GHOST_WINDOW_FRAMES = 60


class TrackMemoryManager:
    """Quản lý toàn bộ lifecycle của track memory và ghost pool."""

    def __init__(
        self,
        ttl_frames: int = TRACK_MEMORY_TTL_FRAMES,
        max_tracks: int = TRACK_MEMORY_MAX_SIZE,
        ghost_window: int = GHOST_WINDOW_FRAMES,
        max_ghosts: int = GHOST_MAX_SIZE
    ):
        self.ttl_frames = ttl_frames
        self.max_tracks = max_tracks
        self.ghost_window = ghost_window
        self.max_ghosts = max_ghosts

        self.tracks: Dict[int, dict] = {}   # track_id -> {attributes, smooth_probs, reid_embedding, ...}
        self.ghosts: Dict[int, dict] = {}   # track_id -> {attributes đã evict}

    def get(self, track_id: int) -> Optional[dict]:
        return self.tracks.get(track_id)

    def update(self, track_id: int, data: dict):
        self.tracks[track_id] = data

    def contains(self, track_id: int) -> bool:
        return track_id in self.tracks

    def evict_stale(self, frame_idx: int):
        """Chuyển track cũ vào ghost pool, dọn ghost quá hạn."""
        stale_ids = [
            tid for tid, mem in self.tracks.items()
            if frame_idx - mem.get("last_updated", 0) > self.ttl_frames
        ]
        for tid in stale_ids:
            mem = self.tracks.pop(tid, None)
            if mem:
                mem["crop"] = None  # Giải phóng RAM ảnh crop
                self.ghosts[tid] = mem

        # Cắt nếu vượt max size
        if len(self.tracks) > self.max_tracks:
            by_age = sorted(self.tracks.items(), key=lambda kv: kv[1].get("last_updated", 0))
            for tid, mem in by_age[: len(self.tracks) - self.max_tracks]:
                self.tracks.pop(tid, None)
                mem["crop"] = None
                self.ghosts[tid] = mem

        # Giữ tối đa max_ghosts gần nhất
        if len(self.ghosts) > self.max_ghosts:
            oldest = sorted(self.ghosts.items(), key=lambda kv: kv[1].get("last_updated", 0))
            for tid, _ in oldest[: len(self.ghosts) - self.max_ghosts]:
                self.ghosts.pop(tid, None)

    def get_valid_ghosts(self, frame_idx: int, window_override: int = None) -> dict:
        """Lấy ghost tracks còn trong cửa sổ thời gian."""
        window = window_override or self.ghost_window
        return {
            tid: g for tid, g in self.ghosts.items()
            if frame_idx - g.get("last_updated", 0) <= window
        }

    def pop_ghost(self, ghost_id: int) -> Optional[dict]:
        return self.ghosts.pop(ghost_id, None)

    def clear(self):
        self.tracks.clear()
        self.ghosts.clear()

    def __len__(self):
        return len(self.tracks)
