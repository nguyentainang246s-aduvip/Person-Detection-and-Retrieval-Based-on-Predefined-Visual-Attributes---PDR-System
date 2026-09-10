"""
src/retrieval/pipeline.py
=========================
Pipeline chính: Tích hợp Detection -> Tracking -> Attribute Recognition -> Color Detection -> Matching.

LUỒNG HOẠT ĐỘNG:
    Video Frame (BGR)
          │
          ▼
     Person Tracker (ByteTrack / BoT-SORT) -> List[tracked_objects]
          │
     ┌────┴─────────────────────────────┐
     ▼                                  ▼
Color Detector (HSV + KMeans)   PAR ResNet50 (Gender, Hat, Backpack,...)
     │                                  │
     └──────────────┬───────────────────┘
                    ▼
         Merged Attributes Dict
                    ↓
        Matching Engine (Tính điểm %)
                    ↓
   Phân loại: TARGET FOUND vs OTHERS
     ├──────────────────────────────┐
     ▼                              ▼
Vẽ Bounding Box Xanh            Lưu ảnh Crop, CSV Log
& Nhãn thuộc tính lên Frame     & Xuất Video Kết quả
"""

import os
import gc
import time
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from src.detection.detector import PersonDetector  # noqa: F401
from src.tracking.tracker import PersonTracker
from src.tracking.reid_embedder import ReIDEmbedder
from src.attributes.color_detector import ColorDetector
from src.attributes.par_model import AttributeRecognizer
from src.retrieval.matcher import AttributeMatcher
from src.retrieval.track_memory import TrackMemoryManager
from src.retrieval.result_writer import ResultWriter
from src.database.db import DatabaseManager
from src.utils.config_loader import load_config
from src.utils.video_utils import (
    open_video, read_frame, release_video,
    get_video_writer, frame_to_timestamp,
    crop_person, resize_frame, FPSCounter
)
from src.utils.visualization import (
    draw_person_info, draw_query_panel, draw_fps_and_count
)
from src.utils.logger import get_logger

logger = get_logger("pipeline")


class PersonRetrievalPipeline:
    """
    Pipeline chính kết nối tất cả module trong hệ thống tìm kiếm người:
    Detection (YOLOv8) -> Tracking (ByteTrack / BoT-SORT) -> Re-ID Appearance Embedding ->
    PAR (Attribute Recognition) -> Color Detection -> Attribute Matching -> Output.
    """

    def __init__(
        self,
        yolo_model_path: str = "models/yolo/yolov8n.pt",
        par_weights_path: str = "models/par/par_resnet50.pth",
        device: str = None,
        confidence_threshold: float = 0.4,
        matching_threshold: float = 0.5,
        attribute_update_interval: int = 5,
        tracker_type: str = None,
        enable_reid: bool = True
    ):
        logger.info("Đang khởi tạo toàn bộ các module trong Pipeline...")

        # Đọc cấu hình từ config.yaml (B2: magic numbers -> config)
        self._config = load_config()
        video_cfg = self._config.get("video", {})
        track_cfg = self._config.get("tracking", {})
        pipe_cfg = self._config.get("pipeline", {})
        reid_cfg = self._config.get("reid", {})

        self.skip_n_frames = video_cfg.get("process_every_n_frames", 1)
        self.max_cnn_per_frame = pipe_cfg.get("max_cnn_per_frame", 2)
        self.ema_alpha = pipe_cfg.get("ema_alpha", 0.35)
        self.eviction_interval = pipe_cfg.get("eviction_interval", 50)
        self.ghost_window_frames = reid_cfg.get("ghost_window_frames", pipe_cfg.get("ghost_window_frames", 150))
        self.reid_weight = reid_cfg.get("reid_weight", 0.70)
        self.attr_weight = reid_cfg.get("attr_weight", 0.30)
        self.reid_cosine_threshold = reid_cfg.get("cosine_threshold", 0.50)
        self.reid_total_threshold = reid_cfg.get("total_score_threshold", 0.55)
        self.reid_ema_alpha = reid_cfg.get("ema_alpha", 0.70)

        self.device = device
        self.matching_threshold = matching_threshold
        self.attr_interval = attribute_update_interval

        # Xác định file cấu hình tracker (ưu tiên tham số truyền vào -> config.yaml -> bytetrack.yaml)
        resolved_tracker = tracker_type or track_cfg.get("tracker", "bytetrack.yaml")

        # 1. Khởi tạo Tracker (ByteTrack / BoT-SORT)
        self.tracker = PersonTracker(
            model_path=yolo_model_path,
            tracker_type=resolved_tracker,
            confidence_threshold=confidence_threshold,
            device=device
        )

        # 1.5. Khởi tạo Re-ID Appearance Embedder (Giai đoạn 1)
        self.enable_reid = enable_reid
        self.reid_embedder = ReIDEmbedder(device=device) if enable_reid else None

        # 2. Khởi tạo Color Detector (K-Means HSV / Delta-E / Deep Learning Color Head)
        color_cfg = self._config.get("color", {})
        color_method = color_cfg.get("method", "kmeans_hsv")
        self.color_detector = ColorDetector(
            n_clusters=color_cfg.get("n_clusters", 3),
            method=color_method,
            device=device
        )

        # 3. Khởi tạo PAR Model
        self.par_recognizer = AttributeRecognizer(
            weights_path=par_weights_path,
            device=device
        )

        # 4. Khởi tạo Matching Engine
        self.matcher = AttributeMatcher(default_threshold=matching_threshold)

        # 5. Khởi tạo Database Manager
        self.db = DatabaseManager()

        # Quản lý bộ nhớ Track và Ghost qua TrackMemoryManager (B1)
        self.memory = TrackMemoryManager(
            ttl_frames=pipe_cfg.get("track_ttl_frames", 300),
            max_tracks=pipe_cfg.get("track_max_size", 500),
            ghost_window=self.ghost_window_frames,
            max_ghosts=pipe_cfg.get("ghost_max_size", 50)
        )
        self.track_memory = self.memory.tracks
        self.ghost_tracks = self.memory.ghosts

        logger.info(f"Pipeline đã sẵn sàng hoạt động! (skip_n_frames={self.skip_n_frames})")

    # ── Context Manager giải phóng VRAM (D2) ──────────────────────────
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def cleanup(self):
        """Giải phóng GPU memory và tài nguyên khi kết thúc."""
        self.memory.clear()
        self.track_memory.clear()
        self.ghost_tracks.clear()

        # Giải phóng model khỏi GPU
        if hasattr(self, "par_recognizer") and hasattr(self.par_recognizer, "model") and self.par_recognizer.model is not None:
            if hasattr(self.par_recognizer.model, "cpu"):
                self.par_recognizer.model.cpu()

        if hasattr(self, "reid_embedder") and self.reid_embedder is not None:
            if hasattr(self.reid_embedder, "features") and hasattr(self.reid_embedder.features, "cpu"):
                self.reid_embedder.features.cpu()

        if hasattr(self, "tracker") and hasattr(self.tracker, "model") and self.tracker.model is not None:
            if hasattr(self.tracker.model, "cpu"):
                self.tracker.model.cpu()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Đã giải phóng toàn bộ tài nguyên GPU/RAM thành công.")

    # ── Kế thừa Ghost Track & Re-ID Matching ───────────────────────────
    def _try_inherit_ghost(
        self,
        new_track_id: int,
        frame_idx: int,
        new_attrs: dict = None,
        new_crop: np.ndarray = None
    ) -> bool:
        """
        Track mới xuất hiện -> kế thừa smooth_probs và thuộc tính từ ghost phù hợp nhất.
        Tích hợp Re-ID Embedding + Đa thuộc tính (C5) + Cửa sổ mở rộng 150 frames (C4).
        """
        if not self.ghost_tracks:
            return False

        # C4: Cửa sổ mở rộng 150 frames
        valid_ghosts = self.memory.get_valid_ghosts(frame_idx, window_override=self.ghost_window_frames)
        if not valid_ghosts:
            return False

        # Trích xuất embedding diện mạo của crop mới
        new_emb = None
        if self.reid_embedder is not None and new_crop is not None and new_crop.size > 0:
            new_emb = self.reid_embedder.extract(new_crop)

        scored = []
        for tid, g in valid_ghosts.items():
            g_attrs = g.get("attributes", {})
            g_emb = g.get("reid_embedding")

            # Điểm tương đồng Re-ID (Cosine Similarity)
            cos_sim = 0.0
            has_reid = (new_emb is not None and g_emb is not None)
            if has_reid:
                cos_sim = self.reid_embedder.cosine_similarity(new_emb, g_emb)

            # C5: Điểm tương đồng thuộc tính mở rộng (4 thuộc tính)
            attr_sim = 0.0
            max_attr_score = 0.0

            # Giới tính (2.0)
            max_attr_score += 2.0
            if new_attrs and g_attrs.get("gender") == new_attrs.get("gender"):
                attr_sim += 2.0

            # Màu áo (1.5)
            max_attr_score += 1.5
            if new_attrs and g_attrs.get("upper_color") == new_attrs.get("upper_color"):
                attr_sim += 1.5

            # Màu quần (1.0)
            max_attr_score += 1.0
            if new_attrs and g_attrs.get("lower_color") == new_attrs.get("lower_color"):
                attr_sim += 1.0

            # Balo (0.5)
            max_attr_score += 0.5
            if new_attrs and g_attrs.get("backpack") == new_attrs.get("backpack"):
                attr_sim += 0.5

            attr_sim_norm = (attr_sim / max_attr_score) if max_attr_score > 0 else 0.0

            # B2: Điểm kết hợp theo trọng số config
            if has_reid:
                total_score = self.reid_weight * max(0.0, cos_sim) + self.attr_weight * attr_sim_norm
            elif new_attrs:
                total_score = attr_sim_norm
            else:
                total_score = 0.50

            scored.append((total_score, cos_sim, g.get("last_updated", 0), tid))

        scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        best_score, best_cos_sim, _, best_id = scored[0]

        # Kiểm tra ngưỡng tin cậy theo cấu hình
        target_ghost = valid_ghosts[best_id]
        if new_emb is not None and target_ghost.get("reid_embedding") is not None:
            if best_cos_sim < self.reid_cosine_threshold and best_score < self.reid_total_threshold:
                return False
        elif new_attrs and best_score < 0.50:
            return False

        ghost = self.ghost_tracks[best_id]
        inherited_emb = ghost.get("reid_embedding")
        if new_emb is not None and inherited_emb is not None:
            inherited_emb = self.reid_embedder.update_moving_average(inherited_emb, new_emb, alpha=self.reid_ema_alpha)
        elif new_emb is not None:
            inherited_emb = new_emb

        self.track_memory[new_track_id] = {
            **ghost,
            "last_updated": frame_idx,
            "inherited_from": best_id,
            "reid_embedding": inherited_emb,
            "crop": None,
        }
        self.memory.pop_ghost(best_id)
        logger.info(f"Re-ID Ghép Track: #{new_track_id} kế thừa từ Ghost #{best_id} (Cosine Sim: {best_cos_sim:.3f}, Score: {best_score:.3f})")
        return True

    def reset(self):
        """Reset toàn bộ trạng thái tracking và memory cho phiên tìm kiếm mới (A1)."""
        self.memory.clear()
        self.track_memory = self.memory.tracks
        self.ghost_tracks = self.memory.ghosts

        if hasattr(self, "last_tracked_objects"):
            if isinstance(self.last_tracked_objects, dict):
                self.last_tracked_objects.clear()
            else:
                self.last_tracked_objects = []

        self.tracker.reset()
        logger.info("Đã reset toàn bộ trạng thái Pipeline thành công.")

    def _evict_stale_tracks(self, frame_idx: int):
        """Xóa các track_id không được cập nhật; chuyển vào ghost_tracks để hỗ trợ kế thừa (B1)."""
        self.memory.evict_stale(frame_idx)

    # ── Xử lý từng Frame ───────────────────────────────────────────────
    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        target_query: dict,
        threshold: float = None
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Xử lý 1 frame video:
          1. Tracking (ByteTrack / BoT-SORT)
          2. Batch inference PAR + Color
          3. Attribute Smoothing (C2: EMA alpha riêng biệt)
          4. Matching Engine
          5. B4: Tối ưu frame.copy()
        """
        thresh = threshold or self.matching_threshold

        # Dọn dẹp định kỳ (B2: eviction_interval từ config)
        if frame_idx % self.eviction_interval == 0:
            self._evict_stale_tracks(frame_idx)

        tracked_objects = self.tracker.track(frame, persist=True)
        self.last_tracked_objects = tracked_objects

        matched_persons_this_frame = []
        # B4: Tối ưu bộ nhớ - chỉ copy khi thực sự có đối tượng cần vẽ
        annotated_frame = frame.copy() if tracked_objects else frame

        # --- Bước 1: xác định object nào CẦN phân tích, xếp theo mức độ "đói" ---
        candidates = []  # (wait_time, obj)
        for obj in tracked_objects:
            track_id = obj["track_id"]
            mem = self.track_memory.get(track_id)
            if mem is None:
                # Trích xuất crop sơ bộ để Re-ID nhận dạng diện mạo kế thừa ghost
                crop_cand = crop_person(frame, obj["bbox"])
                if self._try_inherit_ghost(track_id, frame_idx, new_crop=crop_cand):
                    mem = self.track_memory.get(track_id)
            if mem is None:
                candidates.append((float("inf"), obj))  # track mới toanh -> ưu tiên tuyệt đối
                continue
            obs_count = mem.get("obs_count", 0)
            interval = self.attr_interval if obs_count < 3 else 20
            wait = frame_idx - mem.get("last_updated", 0)
            if wait >= interval:
                candidates.append((wait, obj))

        # Ưu tiên track chờ lâu nhất trước
        candidates.sort(key=lambda x: x[0], reverse=True)
        to_analyze = candidates[:self.max_cnn_per_frame]
        to_analyze_ids = {obj["track_id"] for _, obj in to_analyze}

        # --- Bước 2: crop + batch inference cho các track được chọn ---
        if to_analyze:
            crops, valid_objs = [], []
            for _, obj in to_analyze:
                crop = crop_person(frame, obj["bbox"])
                if crop is not None and crop.size > 0:
                    crops.append(crop)
                    valid_objs.append(obj)

            if crops:
                # 1 forward pass duy nhất cho tất cả crop cần phân tích trong frame này
                par_batch = self.par_recognizer.predict_batch(crops)
                colors_batch = [self.color_detector.detect_colors(c) for c in crops]

                # C2: EMA alpha riêng biệt theo đặc tính từng thuộc tính
                ema_alphas = {
                    "female": 0.20,     # Giới tính ổn định -> tin tưởng lịch sử
                    "hat": 0.45,        # Mũ thay đổi khi xoay đầu -> tin tưởng quan sát mới
                    "glasses": 0.35,    # Kính tương đối ổn định
                    "backpack": 0.25,   # Balo ổn định
                }

                for obj, crop, par_attrs, colors in zip(valid_objs, crops, par_batch, colors_batch):
                    track_id = obj["track_id"]
                    raw = par_attrs.get("raw_probs", {})
                    prev = self.track_memory.get(track_id)

                    if prev and "smooth_probs" in prev:
                        old_probs = prev["smooth_probs"]
                        smooth = {
                            k: (1.0 - ema_alphas[k]) * old_probs[k] + ema_alphas[k] * raw.get(k, old_probs[k])
                            for k in ("female", "hat", "glasses", "backpack")
                        }
                    else:
                        smooth = {
                            "female": raw.get("female", 0.5),
                            "hat": raw.get("hat", 0.0),
                            "glasses": raw.get("glasses", 0.0),
                            "backpack": raw.get("backpack", 0.0),
                        }

                    resolved_attrs = self.par_recognizer.resolve_attributes(smooth)
                    merged_attrs = {**resolved_attrs, **colors}

                    # Trích xuất và cập nhật Re-ID embedding
                    curr_emb = prev.get("reid_embedding") if prev else None
                    if self.reid_embedder is not None and crop is not None:
                        new_emb = self.reid_embedder.extract(crop)
                        if curr_emb is not None:
                            curr_emb = self.reid_embedder.update_moving_average(curr_emb, new_emb, alpha=self.reid_ema_alpha)
                        else:
                            curr_emb = new_emb

                    self.track_memory[track_id] = {
                        "attributes": merged_attrs,
                        "smooth_probs": smooth,
                        "reid_embedding": curr_emb,
                        "crop": crop,
                        "last_updated": frame_idx,
                        "obs_count": (prev.get("obs_count", 0) + 1) if prev else 1,
                    }

        # --- Bước 3: So khớp và Vẽ Annotation ---
        for obj in tracked_objects:
            track_id = obj["track_id"]
            bbox = obj["bbox"]
            mem = self.track_memory.get(track_id)

            if mem is None:
                # Track chưa phân tích xong
                if annotated_frame is not frame:
                    annotated_frame = draw_person_info(
                        annotated_frame, bbox, track_id,
                        attributes={"status": "Dang phan tich..."},
                        score=0.0,
                        is_matched=False
                    )
                continue

            person_attrs = mem["attributes"]
            is_matched, score, breakdown = self.matcher.is_match(
                target_query, person_attrs, threshold=thresh
            )

            # Vẽ thông tin người lên frame
            annotated_frame = draw_person_info(
                annotated_frame,
                bbox=bbox,
                track_id=track_id,
                attributes=person_attrs,
                score=score,
                is_matched=is_matched
            )

            if is_matched:
                matched_persons_this_frame.append({
                    "track_id": track_id,
                    "bbox": bbox,
                    "score": score,
                    "attributes": person_attrs,
                    "breakdown": breakdown,
                    "crop": mem.get("crop")
                })

        return annotated_frame, matched_persons_this_frame

    # ── Xử lý toàn bộ Video ───────────────────────────────────────────
    def run_on_video(
        self,
        video_path: str,
        target_query: dict,
        output_video_path: str = None,
        save_crops_dir: str = "results/crops",
        save_csv_log: str = "results/logs/retrieval_results.csv",
        query_id: int = 1,
        max_frames: int = None,
        display: bool = False,
        matching_threshold: float = None
    ) -> dict:
        """
        Chạy toàn bộ quy trình tìm kiếm trên một file video.
        B1: Tích hợp ResultWriter quản lý I/O.
        B3: Tích hợp Adaptive Frame Skipping.
        """
        thresh = matching_threshold or self.matching_threshold
        logger.info(f"Bắt đầu xử lý video: {video_path}")
        logger.info(f"Mục tiêu tìm kiếm: {target_query} | Ngưỡng điểm: {thresh*100:.0f}%")

        self.reset()
        cap, video_info = open_video(video_path)
        fps = video_info["fps"] or 25.0
        total_frames = video_info["total_frames"]

        writer = None
        if output_video_path:
            os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
            writer = get_video_writer(
                output_video_path,
                fps=fps,
                width=video_info["width"],
                height=video_info["height"]
            )

        # B1: Quản lý CSV và Crops qua ResultWriter
        writer_mgr = ResultWriter(csv_path=save_csv_log, crops_dir=save_crops_dir)

        fps_counter = FPSCounter()
        frame_idx = 0
        all_unique_targets_found = {}

        try:
            while True:
                if max_frames and frame_idx >= max_frames:
                    break

                fps_counter.start_frame()
                success, frame = read_frame(cap)
                if not success:
                    break

                # Frame skipping theo config
                if self.skip_n_frames > 1 and frame_idx % self.skip_n_frames != 0:
                    frame_idx += 1
                    fps_counter.end_frame()
                    continue

                # Xử lý frame qua pipeline
                annotated_frame, targets = self.process_frame(
                    frame, frame_idx, target_query, threshold=thresh
                )

                # Vẽ Query panel và FPS đếm
                annotated_frame = draw_query_panel(annotated_frame, target_query, thresh)
                annotated_frame = draw_fps_and_count(
                    annotated_frame,
                    fps=fps_counter.get_fps(),
                    total_detected=len(self.track_memory),
                    total_matched=len(all_unique_targets_found)
                )

                # Ghi nhận các target tìm thấy
                timestamp = frame_to_timestamp(frame_idx, fps)
                for t in targets:
                    tid = t["track_id"]
                    if tid not in all_unique_targets_found or t["score"] > all_unique_targets_found[tid]["score"]:
                        crop_full_path = writer_mgr.save_crop(t["crop"], tid, timestamp)
                        if tid in self.track_memory:
                            self.track_memory[tid]["crop"] = None

                        all_unique_targets_found[tid] = {
                            "track_id": tid,
                            "score": t["score"],
                            "timestamp": timestamp,
                            "frame_idx": frame_idx,
                            "attributes": t["attributes"],
                            "crop_path": crop_full_path
                        }

                        # Lưu vào SQLite Database
                        self.db.save_target_result(query_id, all_unique_targets_found[tid])

                        if save_csv_log:
                            writer_mgr.write_csv_row(timestamp, frame_idx, tid, t["score"], t["attributes"], crop_full_path)

                if writer:
                    writer.write(annotated_frame)

                if display:
                    disp = resize_frame(annotated_frame, width=1280)
                    cv2.imshow("Person Retrieval Demo", disp)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                fps_counter.end_frame()

                # B3: Adaptive frame skipping tự điều chỉnh theo FPS thực tế
                current_fps = fps_counter.get_fps()
                if current_fps > 0:
                    if current_fps < 10 and self.skip_n_frames < 4:
                        self.skip_n_frames += 1
                        logger.debug(f"FPS thấp ({current_fps:.1f}) -> tăng skip lên {self.skip_n_frames}")
                    elif current_fps > 25 and self.skip_n_frames > 1:
                        self.skip_n_frames -= 1
                        logger.debug(f"FPS cao ({current_fps:.1f}) -> giảm skip xuống {self.skip_n_frames}")

                frame_idx += 1

        finally:
            release_video(cap)
            if writer:
                writer.release()
            writer_mgr.close()
            if display:
                cv2.destroyAllWindows()
            logger.info(f"Đã giải phóng tài nguyên video. Frames đã xử lý: {frame_idx}")

        logger.info(f"Hoàn thành xử lý {frame_idx} frames. Tổng số đối tượng Target tìm thấy: {len(all_unique_targets_found)}")

        return {
            "total_frames_processed": frame_idx,
            "total_tracked_persons": len(self.track_memory),
            "targets_found_count": len(all_unique_targets_found),
            "targets": list(all_unique_targets_found.values()),
            "output_video": output_video_path,
            "csv_log": save_csv_log
        }
