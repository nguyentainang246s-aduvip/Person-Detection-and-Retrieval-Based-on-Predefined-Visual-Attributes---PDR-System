"""
src/retrieval/pipeline.py
=========================
Pipeline tích hợp toàn diện (End-to-End Person Retrieval Pipeline).

LUỒNG XỬ LÝ HỆ THỐNG:
    Video Input (File / Camera)
         ↓
    YOLOv8 Detection (Phát hiện bbox người)
         ↓
    ByteTrack (Gán và duy trì Track ID)
         ↓
    Person Crop (Trích xuất vùng ảnh từng người)
         ├──────────────────────────────┐
         ▼                              ▼
    Color Detector (HSV + KMeans)   PAR ResNet50 (Gender, Hat, Backpack,...)
         │                              │
         └──────────────┬───────────────┘
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
import time
import csv
import cv2
import numpy as np
from pathlib import Path

# PersonDetector: dùng cho standalone scripts (benchmark_fps.py, test_detection.py)
# Không sử dụng trực tiếp trong pipeline production — detect+track đều qua PersonTracker.track()
from src.detection.detector import PersonDetector  # noqa: F401
from src.tracking.tracker import PersonTracker
from src.attributes.color_detector import ColorDetector
from src.attributes.par_model import AttributeRecognizer
from src.retrieval.matcher import AttributeMatcher
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


TRACK_MEMORY_TTL_FRAMES = 300      # ~10-12s ở 25-30fps: track vắng mặt lâu hơn mức này sẽ chuyển sang ghost
TRACK_MEMORY_MAX_SIZE = 500        # chặn trên tuyệt đối, phòng trường hợp fps rất cao / video rất dài


class PersonRetrievalPipeline:
    """
    Bộ điều phối toàn diện cho hệ thống phát hiện và tìm người theo đặc điểm nhận dạng.

    GHI CHÚ KIẾN TRÚC (QUAN TRỌNG):
        - Single-session: Mỗi instance giữ state riêng (tracker, track_memory, ghost_tracks).
          Nếu dùng @st.cache_resource trong Streamlit, MỌI user session sẽ share cùng 1 instance
          → dữ liệu tìm kiếm bị lẫn giữa các user. Với triển khai đa người dùng, cần tạo
          pipeline theo st.session_state thay vì cache_resource toàn cục.
        - Ghost track inheritance hiện chỉ dùng so khớp thuộc tính cơ bản (gender + upper_color),
          CHƯA có Re-ID embedding (appearance vector). Trong tình huống nhiều người occlusion
          đồng thời, vẫn có rủi ro gán nhầm thuộc tính — hướng khắc phục triệt để cần
          tích hợp Re-ID backbone (OSNet/BoT-SORT) để so khớp bằng cosine similarity.
    """

    def __init__(
        self,
        yolo_model_path: str = "models/yolo/yolov8n.pt",
        par_weights_path: str = "models/par/par_resnet50.pth",
        device: str = None,
        confidence_threshold: float = 0.4,
        matching_threshold: float = 0.5,
        attribute_update_interval: int = 5 # Tối ưu hóa: Phân tích lại thuộc tính mỗi 5 frames cho mỗi ID
    ):
        logger.info("Đang khởi tạo toàn bộ các module trong Pipeline...")

        # Đọc cấu hình từ config.yaml
        self._config = load_config()
        video_cfg = self._config.get("video", {})
        self.skip_n_frames = video_cfg.get("process_every_n_frames", 1)

        self.device = device
        self.matching_threshold = matching_threshold
        self.attr_interval = attribute_update_interval

        # 1. Khởi tạo Tracker (ByteTrack/BoT-SORT + YOLO)
        self.tracker = PersonTracker(
            model_path=yolo_model_path,
            confidence_threshold=confidence_threshold,
            device=device
        )

        # 2. Khởi tạo Color Detector
        self.color_detector = ColorDetector(n_clusters=3)

        # 3. Khởi tạo PAR Model (ResNet50)
        self.par_recognizer = AttributeRecognizer(
            weights_path=par_weights_path,
            device=device
        )

        # 4. Khởi tạo Matching Engine
        self.matcher = AttributeMatcher(default_threshold=matching_threshold)

        # 5. Khởi tạo Database Manager
        self.db = DatabaseManager()

        # Bộ nhớ đệm lưu thuộc tính của từng Track ID
        self.track_memory = {}
        # Ghost tracks: lưu các track vừa bị evict gần đây để hỗ trợ kế thừa khi bị ID-switch
        self.ghost_tracks = {}

        logger.info(f"Pipeline đã sẵn sàng hoạt động! (skip_n_frames={self.skip_n_frames})")

    def reset(self):
        """Reset trạng thái bộ nhớ cho lượt tìm kiếm mới."""
        self.tracker.reset()
        self.track_memory.clear()
        self.ghost_tracks.clear()

    def _try_inherit_ghost(self, new_track_id: int, frame_idx: int,
                           new_attrs: dict = None) -> bool:
        """
        Track mới xuất hiện → thử kế thừa smooth_probs từ ghost phù hợp nhất (< 60 frames ~ 2s).

        GIỚI HẠN: So khớp dựa trên thuộc tính cơ bản (gender + upper_color), KHÔNG dùng
        Re-ID embedding (appearance vector). Trong tình huống nhiều người có thuộc tính giống
        nhau bị occlusion đồng thời, vẫn có thể kế thừa nhầm. Hướng khắc phục triệt để:
        tích hợp Re-ID backbone (OSNet) để so khớp bằng cosine similarity.
        """
        if not self.ghost_tracks:
            return False

        # Lọc ghost còn trong cửa sổ thời gian (< 60 frames ~ 2 giây)
        valid_ghosts = {
            tid: g for tid, g in self.ghost_tracks.items()
            if frame_idx - g["last_updated"] <= 60
        }
        if not valid_ghosts:
            return False

        # Nếu có thuộc tính mới (từ frame hiện tại), ưu tiên ghost khớp gender + upper_color
        if new_attrs:
            scored = []
            for tid, g in valid_ghosts.items():
                g_attrs = g.get("attributes", {})
                similarity = 0
                # So khớp gender
                if g_attrs.get("gender") == new_attrs.get("gender"):
                    similarity += 2  # trọng số cao vì gender ít thay đổi
                # So khớp upper_color
                if g_attrs.get("upper_color") == new_attrs.get("upper_color"):
                    similarity += 1
                scored.append((similarity, g["last_updated"], tid))
            # Ưu tiên: khớp thuộc tính nhiều nhất → gần nhất về thời gian
            scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
            best_id = scored[0][2]
        else:
            # Fallback: chọn ghost gần nhất (hành vi cũ, khi chưa có attrs)
            best_id = max(valid_ghosts, key=lambda tid: valid_ghosts[tid]["last_updated"])

        ghost = self.ghost_tracks[best_id]
        self.track_memory[new_track_id] = {
            **ghost,
            "last_updated": frame_idx,
            "inherited_from": best_id,
            "crop": None,
        }
        self.ghost_tracks.pop(best_id, None)
        return True

    def _evict_stale_tracks(self, frame_idx: int):
        """Xóa các track_id không được cập nhật; chuyển vào ghost_tracks để hỗ trợ kế thừa."""
        stale_ids = [
            tid for tid, mem in self.track_memory.items()
            if frame_idx - mem["last_updated"] > TRACK_MEMORY_TTL_FRAMES
        ]
        for tid in stale_ids:
            mem = self.track_memory.pop(tid, None)
            if mem:
                mem["crop"] = None
                self.ghost_tracks[tid] = mem

        if len(self.track_memory) > TRACK_MEMORY_MAX_SIZE:
            by_age = sorted(self.track_memory.items(), key=lambda kv: kv[1]["last_updated"])
            n_to_drop = len(self.track_memory) - TRACK_MEMORY_MAX_SIZE
            for tid, mem in by_age[:n_to_drop]:
                self.track_memory.pop(tid, None)
                mem["crop"] = None
                self.ghost_tracks[tid] = mem

        # Giữ tối đa 50 ghost gần nhất
        if len(self.ghost_tracks) > 50:
            oldest_ghosts = sorted(self.ghost_tracks.items(), key=lambda kv: kv[1]["last_updated"])
            for tid, _ in oldest_ghosts[: len(self.ghost_tracks) - 50]:
                self.ghost_tracks.pop(tid, None)

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        target_query: dict,
        threshold: float = None
    ) -> tuple:
        thresh = threshold or self.matching_threshold

        # Dọn dẹp định kỳ (rẻ, chỉ duyệt dict) để tránh memory leak trên video dài
        if frame_idx % 50 == 0:
            self._evict_stale_tracks(frame_idx)

        tracked_objects = self.tracker.track(frame, persist=True)

        matched_persons_this_frame = []
        annotated_frame = frame.copy()

        MAX_CNN_PER_FRAME = 2

        # --- Bước 1: xác định object nào CẦN phân tích, xếp theo mức độ "đói" ---
        candidates = []  # (wait_time, obj)
        for obj in tracked_objects:
            track_id = obj["track_id"]
            mem = self.track_memory.get(track_id)
            if mem is None:
                # Thử kế thừa thuộc tính từ ghost track khi bị đổi ID
                if self._try_inherit_ghost(track_id, frame_idx):
                    mem = self.track_memory.get(track_id)
            if mem is None:
                candidates.append((float("inf"), obj))  # track mới toanh -> ưu tiên tuyệt đối
                continue
            obs_count = mem.get("obs_count", 0)
            interval = self.attr_interval if obs_count < 3 else 20
            wait = frame_idx - mem["last_updated"]
            if wait >= interval:
                candidates.append((wait, obj))

        # Ưu tiên track chờ lâu nhất trước (thay vì thứ tự trả về của tracker)
        candidates.sort(key=lambda x: x[0], reverse=True)
        to_analyze = candidates[:MAX_CNN_PER_FRAME]
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
                par_batch = self.par_recognizer.predict_batch(crops)   # list[dict]
                colors_batch = [self.color_detector.detect_colors(c) for c in crops]  # rẻ, không cần batch

                for obj, crop, par_attrs, colors in zip(valid_objs, crops, par_batch, colors_batch):
                    track_id = obj["track_id"]
                    raw = par_attrs.get("raw_probs", {})
                    prev = self.track_memory.get(track_id)

                    if prev and "smooth_probs" in prev:
                        old_probs = prev["smooth_probs"]
                        alpha = 0.35
                        smooth = {
                            k: (1.0 - alpha) * old_probs[k] + alpha * raw.get(k, old_probs[k])
                            for k in ("female", "hat", "glasses", "backpack")
                        }
                    else:
                        smooth = {
                            "female": raw.get("female", 0.5),
                            "hat": raw.get("hat", 0.0),
                            "glasses": raw.get("glasses", 0.0),
                            "backpack": raw.get("backpack", 0.0),
                        }

                    is_female = smooth["female"] >= self.par_recognizer.thresholds["gender"]
                    attrs = {
                        "gender": "Female" if is_female else "Male",
                        "gender_confidence": round(smooth["female"] if is_female else 1.0 - smooth["female"], 3),
                        "hat": smooth["hat"] >= self.par_recognizer.thresholds["hat"],
                        "hat_confidence": round(smooth["hat"], 3),
                        "glasses": smooth["glasses"] >= self.par_recognizer.thresholds["glasses"],
                        "glasses_confidence": round(smooth["glasses"], 3),
                        "backpack": smooth["backpack"] >= self.par_recognizer.thresholds["backpack"],
                        "backpack_confidence": round(smooth["backpack"], 3),
                        **colors,
                        "track_id": track_id,
                    }

                    is_match, score, breakdown = self.matcher.is_match(target_query, attrs, threshold=thresh)
                    obs_count = (prev.get("obs_count", 0) + 1) if prev else 1

                    # Lưu thumbnail nhỏ (96x192) trong RAM thay vì giữ ảnh full resolution
                    thumb = cv2.resize(crop, (96, 192)) if (is_match and crop is not None and crop.size > 0) else None

                    self.track_memory[track_id] = {
                        "attributes": attrs,
                        "smooth_probs": smooth,
                        "last_updated": frame_idx,
                        "obs_count": obs_count,
                        "matched": is_match,
                        "score": score,
                        "breakdown": breakdown,
                        "crop": thumb,
                    }

        # --- Bước 3: vẽ + gom kết quả (đọc từ cache, không phân tích lại) ---
        for obj in tracked_objects:
            track_id = obj["track_id"]
            bbox = obj["bbox"]
            mem = self.track_memory.get(track_id)
            if not mem:
                continue

            is_target = mem["matched"]
            info_to_draw = {**mem["attributes"], "score": mem["score"]}

            if is_target:
                matched_persons_this_frame.append({
                    "track_id": track_id,
                    "bbox": bbox,
                    "score": mem["score"],
                    "attributes": mem["attributes"],
                    "crop": mem.get("crop"),
                })

            annotated_frame = draw_person_info(annotated_frame, bbox, info_to_draw, matched=is_target)

        return annotated_frame, matched_persons_this_frame

    def run_on_video(
        self,
        video_source,
        target_query: dict,
        output_video_path: str = None,
        save_crops_dir: str = "results/crops",
        save_csv_log: str = "results/logs/retrieval_results.csv",
        threshold: float = None,
        max_frames: int = None,
        display: bool = False
    ) -> dict:
        """
        Chạy toàn bộ pipeline tìm kiếm trên 1 file video hoặc camera stream.

        OUTPUT:
            summary (dict): Thống kê toàn bộ kết quả tìm kiếm.
        """
        self.reset()
        thresh = threshold or self.matching_threshold

        logger.info(f"Bắt đầu tìm kiếm trên video: {video_source}")
        logger.info(f"Query tìm kiếm: {target_query} | Ngưỡng: {thresh*100:.0f}%")

        # Lưu thông tin phiên tìm kiếm vào Database
        query_id = self.db.save_query(target_query, thresh)

        cap, info = open_video(video_source)
        fps = info["fps"] or 25.0

        writer = None
        if output_video_path:
            writer = get_video_writer(output_video_path, fps, info["width"], info["height"])

        os.makedirs(save_crops_dir, exist_ok=True)
        if save_csv_log:
            os.makedirs(os.path.dirname(save_csv_log), exist_ok=True)
            csv_file = open(save_csv_log, "w", newline="", encoding="utf-8")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "timestamp", "frame_idx", "track_id", "matching_score",
                "gender", "upper_color", "lower_color", "hat", "glasses", "backpack", "crop_image_path"
            ])

        fps_counter = FPSCounter()
        frame_idx = 0
        all_unique_targets_found = {} # track_id -> best result dict

        try:
            while True:
                if max_frames and frame_idx >= max_frames:
                    break

                fps_counter.start_frame()
                success, frame = read_frame(cap)
                if not success:
                    break

                # Frame skipping theo config (process_every_n_frames)
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
                        # Lưu ảnh crop người tìm thấy
                        crop_filename = f"target_track_{tid:03d}_{timestamp.replace(':', '-')}.jpg"
                        crop_full_path = os.path.join(save_crops_dir, crop_filename)
                        if t["crop"] is not None and t["crop"].size > 0:
                            cv2.imwrite(crop_full_path, t["crop"])
                            # Giải phóng ngay crop khỏi RAM
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
                            attr = t["attributes"]
                            csv_writer.writerow([
                                timestamp, frame_idx, tid, f"{t['score']*100:.1f}%",
                                attr.get("gender"), attr.get("upper_color"), attr.get("lower_color"),
                                attr.get("hat"), attr.get("glasses"), attr.get("backpack"), crop_full_path
                            ])
                            csv_file.flush()

                if writer:
                    writer.write(annotated_frame)

                if display:
                    disp = resize_frame(annotated_frame, width=1280)
                    cv2.imshow("Person Retrieval Demo", disp)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                fps_counter.end_frame()
                frame_idx += 1

        finally:
            # Đảm bảo giải phóng TẤT CẢ tài nguyên dù có exception hay không
            release_video(cap)
            if writer:
                writer.release()
            if save_csv_log:
                try:
                    csv_file.close()
                except Exception:
                    pass
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
