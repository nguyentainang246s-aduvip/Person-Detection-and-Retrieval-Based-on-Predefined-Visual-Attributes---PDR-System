"""
src/retrieval/global_gallery.py
===============================
Bộ nhớ toàn cục lưu trữ Re-ID embedding từ TẤT CẢ camera trong hệ thống.
Cho phép nhận diện và liên kết cùng một người khi di chuyển qua các góc camera khác nhau (Cross-Camera Re-ID).
"""

import threading
import numpy as np
from collections import defaultdict
from typing import Dict, Set, Optional, Tuple
from src.utils.logger import get_logger

logger = get_logger("global_gallery")


class GlobalReIDGallery:
    """
    Thread-safe gallery lưu embedding của tất cả track từ mọi camera.
    Khi track mới xuất hiện trên camera B, so khớp với gallery toàn cục
    để tìm xem đó có phải người đã thấy trên camera A không.
    """

    def __init__(self, similarity_threshold: float = 0.65, max_entries: int = 1000):
        self.lock = threading.Lock()
        self.gallery: Dict[int, dict] = {}  # global_id -> {"embedding": np.ndarray, "cameras": set, "attrs": dict}
        self.camera_track_map: Dict[str, Dict[int, int]] = defaultdict(dict)  # camera_id -> {local_track_id: global_id}
        self.next_global_id = 1
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries

    def query_or_register(
        self,
        camera_id: str,
        local_track_id: int,
        embedding: np.ndarray,
        attributes: dict = None
    ) -> Tuple[int, bool]:
        """
        Trả về (global_id, is_new):
        - Nếu tìm thấy match trong gallery -> trả về global_id đã tồn tại, is_new=False
        - Nếu không -> đăng ký mới và trả về global_id mới, is_new=True
        """
        if embedding is None or embedding.size == 0:
            return local_track_id, False

        with self.lock:
            # Đã ánh xạ trong camera này trước đó chưa?
            existing_gid = self.camera_track_map[camera_id].get(local_track_id)
            if existing_gid and existing_gid in self.gallery:
                # Cập nhật embedding mượt mà (Moving average)
                old_emb = self.gallery[existing_gid]["embedding"]
                updated = 0.8 * old_emb + 0.2 * embedding
                norm = np.linalg.norm(updated)
                if norm > 1e-6:
                    self.gallery[existing_gid]["embedding"] = updated / norm
                return existing_gid, False

            # So khớp với toàn bộ gallery hiện có
            best_gid, best_sim = None, 0.0
            for gid, entry in self.gallery.items():
                sim = float(np.dot(embedding, entry["embedding"]))
                if sim > best_sim:
                    best_sim = sim
                    best_gid = gid

            if best_sim >= self.similarity_threshold and best_gid is not None:
                # Đã tìm thấy khớp xuyên camera (Cross-Camera Match!)
                self.gallery[best_gid]["cameras"].add(camera_id)
                self.camera_track_map[camera_id][local_track_id] = best_gid
                logger.info(
                    f"Cross-camera match: Camera [{camera_id}] Track #{local_track_id} "
                    f"== Global Person #{best_gid} (Cosine Sim={best_sim:.3f}, Cameras={self.gallery[best_gid]['cameras']})"
                )
                return best_gid, False

            # Chưa có trong gallery -> Đăng ký định danh toàn cục mới
            gid = self.next_global_id
            self.next_global_id += 1
            norm = np.linalg.norm(embedding)
            norm_emb = (embedding / norm) if norm > 1e-6 else embedding

            self.gallery[gid] = {
                "embedding": norm_emb.copy(),
                "cameras": {camera_id},
                "attrs": attributes or {},
            }
            self.camera_track_map[camera_id][local_track_id] = gid

            # Eviction nếu vượt quá dung lượng tối đa
            if len(self.gallery) > self.max_entries:
                oldest_gid = min(self.gallery.keys())
                self.gallery.pop(oldest_gid, None)

            return gid, True

    def clear(self):
        with self.lock:
            self.gallery.clear()
            self.camera_track_map.clear()
            self.next_global_id = 1
