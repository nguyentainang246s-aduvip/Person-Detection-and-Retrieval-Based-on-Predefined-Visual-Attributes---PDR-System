"""
src/utils/video_utils.py
========================
Các hàm tiện ích để đọc, xử lý và ghi video.

LUỒNG XỬ LÝ:
    Video File / Webcam
         ↓
    VideoCapture (OpenCV)
         ↓
    Từng Frame (numpy array BGR)
         ↓
    Xử lý (detection, tracking, ...)
         ↓
    Vẽ annotation lên frame
         ↓
    Ghi ra file / hiển thị

Kiến thức cần biết:
- Frame = một ảnh tĩnh trong video (numpy array, shape: H x W x 3)
- BGR = định dạng màu mặc định của OpenCV (Blue-Green-Red), ngược với RGB
- FPS (Frames Per Second) = số frame trên giây
"""

import cv2
import time
import os
import math
import numpy as np
from pathlib import Path


def open_video(source):
    """
    Mở nguồn video (file hoặc webcam).

    INPUT:
        source (str | int):
            - Đường dẫn file video: "data/test_videos/sample.mp4"
            - Webcam: 0 (camera mặc định), 1 (camera ngoài)

    OUTPUT:
        cap (cv2.VideoCapture): Object đọc video
        info (dict): Thông tin video {width, height, fps, total_frames}

    VÍ DỤ:
        cap, info = open_video("data/test_videos/sample.mp4")
        print(info)  # {'width': 1920, 'height': 1080, 'fps': 30.0, ...}
    """
    if isinstance(source, str) and source.strip().isdigit():
        source = int(source.strip())

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise ValueError(f"Không thể mở video hoặc camera: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 25.0

    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(fps),
        "total_frames": max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))),
        "source": str(source),
    }

    return cap, info


def open_video_robust(source, max_retries: int = 5, backoff: float = 1.5):
    """
    Mở nguồn video với cơ chế retry và exponential backoff (hữu ích cho RTSP / IP camera).
    """
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            cap, info = open_video(source)
            if cap is not None and cap.isOpened():
                return cap, info
        except Exception as e:
            last_err = e

        wait_time = backoff * (2 ** (attempt - 1))
        time.sleep(min(wait_time, 10.0))

    raise ConnectionError(f"Không thể kết nối tới nguồn video '{source}' sau {max_retries} lần thử: {last_err}")


def read_frame(cap):
    """
    Đọc một frame từ video.

    INPUT:
        cap: VideoCapture object (từ open_video)

    OUTPUT:
        success (bool): True nếu đọc được frame, False nếu hết video
        frame (numpy.ndarray | None): Frame ảnh shape (H, W, 3), định dạng BGR
                                      None nếu không đọc được

    VÍ DỤ:
        success, frame = read_frame(cap)
        if success:
            # frame là numpy array (H, W, 3) BGR
            print(frame.shape)  # (1080, 1920, 3)
    """
    success, frame = cap.read()
    if not success:
        return False, None
    return True, frame


def release_video(cap):
    """
    Giải phóng tài nguyên video sau khi xử lý xong.
    Luôn gọi hàm này khi kết thúc.
    """
    if cap is not None:
        cap.release()


def get_video_writer(output_path, fps, width, height):
    """
    Tạo object để ghi video output (có annotation).

    INPUT:
        output_path (str): Đường dẫn file output, ví dụ "results/output.mp4"
        fps (float): FPS của video output
        width (int): Chiều rộng frame
        height (int): Chiều cao frame

    OUTPUT:
        writer (cv2.VideoWriter): Object ghi video

    VÍ DỤ:
        writer = get_video_writer("results/output.mp4", 30.0, 1920, 1080)
        writer.write(annotated_frame)
        writer.release()
    """
    # Đảm bảo thư mục output tồn tại
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Codec mp4v cho file .mp4
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    return writer


def frame_to_timestamp(frame_idx, fps):
    """
    Chuyển số thứ tự frame sang timestamp dạng HH:MM:SS.

    INPUT:
        frame_idx (int): Số thứ tự frame (bắt đầu từ 0)
        fps (float): FPS của video

    OUTPUT:
        str: Timestamp dạng "00:01:32"

    VÍ DỤ:
        ts = frame_to_timestamp(2760, 30.0)
        print(ts)  # "00:01:32"
    """
    total_seconds = int(frame_idx / fps)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def resize_frame(frame, width=None, height=None):
    """
    Resize frame về kích thước muốn, giữ nguyên tỉ lệ.

    INPUT:
        frame: numpy array (H, W, 3)
        width (int | None): Chiều rộng mong muốn
        height (int | None): Chiều cao mong muốn (nếu không có width)

    OUTPUT:
        resized_frame: numpy array đã resize

    VÍ DỤ:
        small = resize_frame(frame, width=640)  # Scale về width=640
    """
    h, w = frame.shape[:2]

    if width is None and height is None:
        return frame

    if width is not None:
        ratio = width / w
        new_size = (width, int(h * ratio))
    else:
        ratio = height / h
        new_size = (int(w * ratio), height)

    return cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)


def crop_person(frame, bbox):
    """
    Cắt vùng người từ frame dựa trên bounding box.

    INPUT:
        frame: numpy array (H, W, 3) - frame gốc
        bbox: [x1, y1, x2, y2] - tọa độ bounding box (pixel)
                x1,y1 = góc trên trái | x2,y2 = góc dưới phải

    OUTPUT:
        crop: numpy array - ảnh crop người, hoặc None nếu bbox không hợp lệ

    VÍ DỤ:
        bbox = [100, 50, 300, 400]  # Người từ pixel (100,50) đến (300,400)
        person_img = crop_person(frame, bbox)
        # person_img.shape = (350, 200, 3)
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]

    # Đảm bảo bbox không vượt ra ngoài frame
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    # Kiểm tra bbox có hợp lệ không (diện tích > 0)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    return crop


def bgr_to_rgb(frame):
    """
    Chuyển đổi frame từ BGR (OpenCV) sang RGB (Streamlit/PIL/matplotlib).

    TẠI SAO CẦN:
    - OpenCV đọc ảnh theo định dạng BGR (Blue-Green-Red)
    - Streamlit và hầu hết thư viện khác dùng RGB (Red-Green-Blue)
    - Nếu không convert, màu sẽ bị sai (đỏ thành xanh và ngược lại)
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class FPSCounter:
    """
    Đếm FPS thực tế khi xử lý video.

    VÍ DỤ:
        fps_counter = FPSCounter()
        while True:
            fps_counter.start_frame()
            # ... xử lý frame ...
            fps = fps_counter.get_fps()
            print(f"FPS: {fps:.1f}")
    """

    def __init__(self, avg_window=30):
        """
        avg_window: Số frame để tính FPS trung bình (làm mượt số liệu)
        """
        self.frame_times = []
        self.avg_window = avg_window
        self._start_time = None

    def start_frame(self):
        """Gọi vào đầu mỗi vòng lặp frame."""
        self._start_time = time.time()

    def end_frame(self):
        """Gọi vào cuối mỗi vòng lặp frame để ghi lại thời gian."""
        if self._start_time is not None:
            elapsed = time.time() - self._start_time
            self.frame_times.append(elapsed)
            # Chỉ giữ avg_window frame gần nhất
            if len(self.frame_times) > self.avg_window:
                self.frame_times.pop(0)

    def get_fps(self):
        """Trả về FPS trung bình hiện tại."""
        if not self.frame_times:
            return 0.0
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0
