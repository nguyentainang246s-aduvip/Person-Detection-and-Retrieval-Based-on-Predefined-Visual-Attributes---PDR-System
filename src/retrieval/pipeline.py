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

from src.detection.detector import PersonDetector
from src.tracking.tracker import PersonTracker
from src.attributes.color_detector import ColorDetector
from src.attributes.par_model import AttributeRecognizer
from src.retrieval.matcher import AttributeMatcher
from src.database.db import DatabaseManager
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
    Bộ điều phối toàn diện cho hệ thống phát hiện và tìm người theo đặc điểm nhận dạng.
    """

    def __init__(
        self,
        yolo_model_path: str = "models/yolo/yolov8n.pt",
        par_weights_path: str = "models/par/par_resnet50.pth",
        device: str = None,
        confidence_threshold: float = 0.4,
        matching_threshold: float = 0.7,
        attribute_update_interval: int = 5 # Tối ưu hóa: Phân tích lại thuộc tính mỗi 5 frames cho mỗi ID
    ):
        logger.info("Đang khởi tạo toàn bộ các module trong Pipeline...")

        self.device = device
        self.matching_threshold = matching_threshold
        self.attr_interval = attribute_update_interval

        # 1. Khởi tạo Tracker (ByteTrack + YOLO)
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

        # Bộ nhớ đệm lưu thuộc tính của từng Track ID (tránh chạy ResNet liên tục mỗi frame trên cùng 1 người)
        # track_id -> {"attributes": dict, "last_updated": int, "matched": bool, "score": float}
        self.track_memory = {}

        logger.info("Pipeline đã sẵn sàng hoạt động!")

    def reset(self):
        """Reset trạng thái bộ nhớ cho lượt tìm kiếm mới."""
        self.tracker.reset()
        self.track_memory.clear()

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        target_query: dict,
        threshold: float = None
    ) -> tuple:
        """
        Xử lý 1 frame video đơn lẻ.

        INPUT:
            frame: Ảnh BGR (H, W, 3)
            frame_idx: Số thứ tự frame
            target_query: Bộ tiêu chí tìm kiếm
            threshold: Ngưỡng matching (mặc định lấy theo pipeline)

        OUTPUT:
            annotated_frame (np.ndarray): Frame đã vẽ bounding box và nhãn
            matched_persons_this_frame (list): Danh sách người khớp query trong frame này
        """
        thresh = threshold or self.matching_threshold

        # 1. Tracking: Lấy danh sách người kèm Track ID
        tracked_objects = self.tracker.track(frame, persist=True)

        matched_persons_this_frame = []
        annotated_frame = frame.copy()

        # Phân phối tải: Giới hạn tối đa 2 người chạy mạng CNN nặng trong cùng 1 frame
        # để đảm bảo tốc độ khung hình (FPS) luôn duy trì mượt mà > 20 FPS
        cnn_analysis_count = 0
        MAX_CNN_PER_FRAME = 2

        for obj in tracked_objects:
            track_id = obj["track_id"]
            bbox = obj["bbox"]

            # Kiểm tra xem Track ID này đã được phân tích chưa
            need_analysis = True
            if track_id in self.track_memory:
                mem = self.track_memory[track_id]
                last_frame = mem["last_updated"]
                obs_count = mem.get("obs_count", 0)
                # Người đã có 3 lần phân tích ổn định thì chỉ cập nhật lại sau mỗi 20 frames
                interval = self.attr_interval if obs_count < 3 else 20
                if frame_idx - last_frame < interval:
                    need_analysis = False

            # Nếu vượt quá ngân sách tính toán trong 1 frame, hoãn phân tích người này sang frame kế tiếp
            if need_analysis and cnn_analysis_count >= MAX_CNN_PER_FRAME:
                need_analysis = False

            if need_analysis:
                # 2. Crop ảnh người
                person_crop = crop_person(frame, bbox)

                if person_crop is not None and person_crop.size > 0:
                    cnn_analysis_count += 1

                    # 3. Nhận dạng màu sắc (Áo, Quần) - Chạy tức thì < 0.1ms
                    colors = self.color_detector.detect_colors(person_crop)

                    # 4. Nhận dạng thuộc tính AI (Giới tính, Mũ, Kính, Balo)
                    par_attrs = self.par_recognizer.predict(person_crop)
                    raw = par_attrs.get("raw_probs", {})

                    # Áp dụng bộ lọc trung bình tích lũy theo thời gian (Temporal EMA Smoothing)
                    # để loại bỏ hoàn toàn rung giật nhãn khi người di chuyển hoặc xoay góc
                    if track_id in self.track_memory and "smooth_probs" in self.track_memory[track_id]:
                        old_probs = self.track_memory[track_id]["smooth_probs"]
                        alpha = 0.35  # 35% frame mới, 65% lịch sử tích lũy
                        smooth_p_female = (1.0 - alpha) * old_probs["female"] + alpha * raw.get("female", 0.5)
                        smooth_p_hat = (1.0 - alpha) * old_probs["hat"] + alpha * raw.get("hat", 0.0)
                        smooth_p_glasses = (1.0 - alpha) * old_probs["glasses"] + alpha * raw.get("glasses", 0.0)
                        smooth_p_backpack = (1.0 - alpha) * old_probs["backpack"] + alpha * raw.get("backpack", 0.0)
                    else:
                        smooth_p_female = raw.get("female", 0.5)
                        smooth_p_hat = raw.get("hat", 0.0)
                        smooth_p_glasses = raw.get("glasses", 0.0)
                        smooth_p_backpack = raw.get("backpack", 0.0)

                    smooth_probs = {
                        "female": smooth_p_female,
                        "hat": smooth_p_hat,
                        "glasses": smooth_p_glasses,
                        "backpack": smooth_p_backpack
                    }

                    # Quyết định thuộc tính sau khi đã làm mịn qua nhiều frame
                    is_female = smooth_p_female >= self.par_recognizer.thresholds["gender"]
                    final_gender = "Female" if is_female else "Male"
                    final_hat = smooth_p_hat >= self.par_recognizer.thresholds["hat"]
                    final_glasses = smooth_p_glasses >= self.par_recognizer.thresholds["glasses"]
                    final_backpack = smooth_p_backpack >= self.par_recognizer.thresholds["backpack"]

                    # 5. Hợp nhất thuộc tính
                    attrs = {
                        "gender": final_gender,
                        "gender_confidence": round(smooth_p_female if is_female else (1.0 - smooth_p_female), 3),
                        "hat": final_hat,
                        "hat_confidence": round(smooth_p_hat, 3),
                        "glasses": final_glasses,
                        "glasses_confidence": round(smooth_p_glasses, 3),
                        "backpack": final_backpack,
                        "backpack_confidence": round(smooth_p_backpack, 3),
                        **colors,
                        "track_id": track_id
                    }

                    # 6. So khớp với Query của người dùng
                    is_match, score, breakdown = self.matcher.is_match(target_query, attrs, threshold=thresh)

                    # Lưu vào bộ nhớ đệm
                    self.track_memory[track_id] = {
                        "attributes": attrs,
                        "smooth_probs": smooth_probs,
                        "last_updated": frame_idx,
                        "matched": is_match,
                        "score": score,
                        "breakdown": breakdown,
                        "crop": person_crop
                    }

            # Lấy thông tin từ bộ nhớ đệm để hiển thị
            mem = self.track_memory.get(track_id)
            if mem:
                is_target = mem["matched"]
                info_to_draw = {**mem["attributes"], "score": mem["score"]}

                if is_target:
                    matched_persons_this_frame.append({
                        "track_id": track_id,
                        "bbox": bbox,
                        "score": mem["score"],
                        "attributes": mem["attributes"],
                        "crop": mem.get("crop")
                    })

                # Vẽ bounding box & nhãn lên frame
                annotated_frame = draw_person_info(
                    annotated_frame,
                    bbox,
                    info_to_draw,
                    matched=is_target
                )

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

        while True:
            if max_frames and frame_idx >= max_frames:
                break

            fps_counter.start_frame()
            success, frame = read_frame(cap)
            if not success:
                break

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

        release_video(cap)
        if writer:
            writer.release()
        if save_csv_log:
            csv_file.close()
        if display:
            cv2.destroyAllWindows()

        logger.info(f"Hoàn thành xử lý {frame_idx} frames. Tổng số đối tượng Target tìm thấy: {len(all_unique_targets_found)}")

        return {
            "total_frames_processed": frame_idx,
            "total_tracked_persons": len(self.track_memory),
            "targets_found_count": len(all_unique_targets_found),
            "targets": list(all_unique_targets_found.values()),
            "output_video": output_video_path,
            "csv_log": save_csv_log
        }
