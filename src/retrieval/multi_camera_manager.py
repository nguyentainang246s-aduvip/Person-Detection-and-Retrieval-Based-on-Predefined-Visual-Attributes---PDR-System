"""
src/retrieval/multi_camera_manager.py
=====================================
Kiến trúc luồng xử lý Multi-Camera sẵn sàng cho Production (Giai đoạn 5).

Tính năng:
  - Ingestion đa luồng: Mỗi camera chạy 1 thread đọc video độc lập với hàng đợi an toàn đa luồng (Thread-safe Queue).
  - Kháng lag (Backpressure handling): Hàng đợi vòng giới hạn (circular buffer/drop oldest frame) ngăn trễ hình khi worker bận.
  - Worker Pool: ThreadPoolExecutor xử lý suy luận song song mà không nghẽn luồng đọc video.
  - Session Isolation: Mỗi camera / phiên làm việc duy trì tracker state và gallery độc lập hoàn toàn.
"""

import os
import time
import queue
import threading
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Callable, Any
import cv2
import numpy as np
import psutil

logger = logging.getLogger(__name__)


class CameraStreamReader:
    """
    Luồng đọc khung hình độc lập cho từng nguồn camera (Video file hoặc RTSP).
    Sử dụng hàng đợi Thread-safe Queue giới hạn dung lượng để tránh tràn RAM và loại bỏ độ trễ tích lũy.
    """

    def __init__(self, camera_id: str, source_path: str, max_queue_size: int = 20):
        self.camera_id = camera_id
        self.source_path = source_path
        self.max_queue_size = max_queue_size
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.total_frames_read = 0
        self.dropped_frames = 0
        self.fps = 0.0

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._read_loop, name=f"CamReader-{self.camera_id}", daemon=True)
        self.thread.start()

    def _read_loop(self):
        is_live = isinstance(self.source_path, str) and (
            self.source_path.startswith("rtsp://") or
            self.source_path.startswith("http://") or
            self.source_path.isdigit()
        )

        cap = cv2.VideoCapture(self.source_path)
        consecutive_fails = 0
        MAX_CONSECUTIVE_FAILS = 30  # ~1 giây ở 30fps

        while self.is_running:
            if not cap.isOpened():
                # Reconnect với exponential backoff
                logger.warning(f"[{self.camera_id}] Mất kết nối, đang thử lại...")
                time.sleep(min(2 ** consecutive_fails, 10))
                cap = cv2.VideoCapture(self.source_path)
                consecutive_fails += 1
                if consecutive_fails > 10:
                    logger.error(f"[{self.camera_id}] Không thể kết nối lại sau 10 lần.")
                    break
                continue

            ret, frame = cap.read()
            if not ret:
                consecutive_fails += 1
                if is_live:
                    if consecutive_fails > MAX_CONSECUTIVE_FAILS:
                        logger.warning(f"[{self.camera_id}] Live stream mất tín hiệu, reconnecting...")
                        cap.release()
                        cap = cv2.VideoCapture(self.source_path)
                    continue
                else:
                    # Video file: loop lại
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

            consecutive_fails = 0  # Reset counter khi đọc thành công
            self.total_frames_read += 1

            if self.frame_queue.full():
                try:
                    # Bỏ frame cũ nhất để tránh lag tích lũy (drop oldest)
                    self.frame_queue.get_nowait()
                    self.dropped_frames += 1
                except queue.Empty:
                    pass

            self.frame_queue.put(frame)
            time.sleep(0.033)

        cap.release()

    def get_latest_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)


class MultiCameraSystem:
    """
    Quản lý đồng thời nhiều Camera Stream và phân phối cho Worker Pool xử lý.
    """

    def __init__(self, max_workers: int = 4, pipeline_factory: Optional[Callable[[], Any]] = None):
        self.max_workers = max_workers
        self.cameras: Dict[str, CameraStreamReader] = {}
        self.pipelines: Dict[str, Any] = {}  # 1 pipeline riêng biệt cho mỗi camera
        self.pipeline_factory = pipeline_factory
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="PDR-Worker")
        self.is_active = False

    def add_camera(self, camera_id: str, source_path: str):
        if camera_id in self.cameras:
            self.cameras[camera_id].stop()
        reader = CameraStreamReader(camera_id, source_path)
        self.cameras[camera_id] = reader

        # Khởi tạo pipeline riêng cho camera này nếu có factory
        if self.pipeline_factory:
            self.pipelines[camera_id] = self.pipeline_factory()

    def get_pipeline(self, camera_id: str) -> Optional[Any]:
        return self.pipelines.get(camera_id)

    def start_all(self):
        self.is_active = True
        for cam in self.cameras.values():
            cam.start()

    def stop_all(self, timeout: float = 5.0):
        """Dừng tất cả camera và chờ worker hoàn thành an toàn."""
        self.is_active = False
        for cam in self.cameras.values():
            cam.stop()
        self.executor.shutdown(wait=True, cancel_futures=True)
        logger.info("Đã dừng toàn bộ camera và worker an toàn.")

    def process_all_concurrently(self, process_fn: Callable[[str, np.ndarray], Any]) -> Dict[str, Any]:
        """
        Lấy frame mới nhất từ tất cả camera và xử lý song song trên ThreadPoolExecutor.
        """
        futures = {}
        for cam_id, reader in self.cameras.items():
            frame = reader.get_latest_frame()
            if frame is not None:
                fut = self.executor.submit(process_fn, cam_id, frame)
                futures[cam_id] = fut

        results = {}
        for cam_id, fut in futures.items():
            try:
                results[cam_id] = fut.result(timeout=1.0)
            except Exception as e:
                logger.warning(f"Lỗi xử lý camera {cam_id}: {e}")
                results[cam_id] = None
        return results
