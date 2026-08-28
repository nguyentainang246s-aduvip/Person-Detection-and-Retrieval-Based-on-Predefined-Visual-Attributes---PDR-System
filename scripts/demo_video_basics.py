"""
scripts/demo_video_basics.py
==============================
BÀI TẬP THỰC HÀNH – PHASE 1

Script này giúp bạn hiểu cách OpenCV xử lý video.
Đây là nền tảng của toàn bộ pipeline sau này.

KIẾN THỨC DEMO:
1. Đọc video frame-by-frame
2. Hiển thị thông tin video
3. Vẽ text lên frame (FPS, frame count)
4. Lưu video output
5. Hiểu khái niệm BGR vs RGB
6. Hiểu numpy array là gì trong context ảnh/video

CÁCH CHẠY:
    .\\venv\\Scripts\\python.exe scripts/demo_video_basics.py --source data/test_videos/your_video.mp4

    Nếu không có video:
    .\\venv\\Scripts\\python.exe scripts/demo_video_basics.py --source 0
    (0 = webcam)

PHÍM TẮT KHI CHẠY:
    Q = Thoát
    P = Pause/Resume
    S = Save current frame as image
"""

import sys
import os
import argparse

# Thêm thư mục gốc vào Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from src.utils.video_utils import (
    open_video, read_frame, release_video,
    get_video_writer, frame_to_timestamp,
    resize_frame, bgr_to_rgb, FPSCounter
)
from src.utils.logger import get_logger

logger = get_logger("demo_video")


def draw_info_overlay(frame, frame_idx, fps, total_frames, video_info):
    """
    Vẽ thông tin lên frame để hiển thị.

    GIẢI THÍCH:
    - cv2.putText: Vẽ text lên ảnh
    - cv2.rectangle: Vẽ hình chữ nhật (dùng làm background cho text)
    - (B, G, R): Màu sắc theo định dạng BGR của OpenCV

    Ví dụ màu:
        (0, 255, 0)   = Xanh lá (Green)
        (0, 0, 255)   = Đỏ (Red)
        (255, 0, 0)   = Xanh dương (Blue)
        (255, 255, 0) = Cyan
        (0, 165, 255) = Cam (Orange)
    """
    h, w = frame.shape[:2]

    # Vẽ nền đen mờ ở góc trên trái cho dễ đọc text
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (400, 110), (0, 0, 0), -1)  # -1 = fill
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)         # Blend để mờ 60%

    # Timestamp
    timestamp = frame_to_timestamp(frame_idx, video_info["fps"])

    # Thông tin hiển thị
    info_lines = [
        f"Frame: {frame_idx:05d} / {total_frames}",
        f"Time: {timestamp}",
        f"FPS: {fps:.1f}",
        f"Size: {w}x{h}",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, line in enumerate(info_lines):
        y_pos = 25 + i * 22
        cv2.putText(frame, line, (10, y_pos), font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

    return frame


def process_video(source, output_path=None, max_frames=None, display=True):
    """
    Pipeline đọc và xử lý video cơ bản.

    INPUT:
        source: Đường dẫn video hoặc 0 (webcam)
        output_path: Nếu có, lưu video kết quả ra đây
        max_frames: Nếu có, chỉ xử lý tối đa N frames (test nhanh)
        display: Có hiển thị cửa sổ preview không

    LUỒNG XỬ LÝ:
        Mở video
           ↓
        Loop từng frame:
           Đọc frame
           → Vẽ thông tin lên frame
           → Hiển thị (nếu display=True)
           → Ghi vào file output (nếu output_path có)
           → Kiểm tra phím bấm
        Kết thúc: Giải phóng tài nguyên
    """
    # ── Bước 1: Mở video ──
    logger.info(f"Đang mở video: {source}")

    # Nếu source là số (webcam), convert sang int
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap, info = open_video(source)

    logger.info(f"Video info: {info['width']}x{info['height']} @ {info['fps']:.1f}fps | "
                f"Total frames: {info['total_frames']}")

    # ── Bước 2: Chuẩn bị video writer (nếu cần lưu) ──
    writer = None
    if output_path:
        writer = get_video_writer(output_path, info["fps"], info["width"], info["height"])
        logger.info(f"Sẽ lưu kết quả vào: {output_path}")

    # ── Bước 3: Khởi tạo FPS counter và biến theo dõi ──
    fps_counter = FPSCounter(avg_window=30)
    frame_idx = 0
    paused = False

    print("\n─────────────────────────────────")
    print("  DEMO VIDEO BASICS")
    print("─────────────────────────────────")
    print(f"  Source : {source}")
    print(f"  Size   : {info['width']}x{info['height']}")
    print(f"  FPS    : {info['fps']:.1f}")
    print(f"  Frames : {info['total_frames']}")
    print("─────────────────────────────────")
    if display:
        print("  Phím: [Q] Thoát | [P] Pause | [S] Lưu frame")
    print("─────────────────────────────────\n")

    # ── Bước 4: Vòng lặp chính ──
    while True:
        # Kiểm tra giới hạn frame
        if max_frames and frame_idx >= max_frames:
            logger.info(f"Đã xử lý {max_frames} frames. Dừng lại.")
            break

        if paused:
            # Khi pause: vẫn lắng nghe phím bấm, không đọc frame mới
            if display:
                key = cv2.waitKey(50) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = False
                    logger.info("▶ Resume")
            continue

        # ── 4.1: Bắt đầu đếm thời gian frame ──
        fps_counter.start_frame()

        # ── 4.2: Đọc frame ──
        success, frame = read_frame(cap)
        if not success:
            logger.info("Hết video.")
            break

        # ── 4.3: Xử lý frame ──
        # Ở Phase 1, chỉ vẽ thông tin lên frame
        # Ở các Phase sau, đây là nơi gọi YOLO, tracker, attribute model
        current_fps = fps_counter.get_fps()
        frame = draw_info_overlay(frame, frame_idx, current_fps, info["total_frames"], info)

        # ── 4.4: Ghi vào file output ──
        if writer:
            writer.write(frame)

        # ── 4.5: Hiển thị ──
        if display:
            # Resize xuống để hiển thị vừa màn hình (không thay đổi file gốc)
            display_frame = resize_frame(frame, width=1280)
            cv2.imshow("Person Retrieval - Phase 1 Demo", display_frame)

            # Xử lý phím bấm (waitKey 1ms để không block)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Người dùng nhấn Q. Thoát.")
                break
            elif key == ord('p'):
                paused = True
                logger.info("⏸ Paused. Nhấn P để tiếp tục.")
            elif key == ord('s'):
                save_path = f"results/frame_{frame_idx:05d}.jpg"
                os.makedirs("results", exist_ok=True)
                cv2.imwrite(save_path, frame)
                logger.info(f"Đã lưu frame: {save_path}")

        # ── 4.6: Kết thúc đếm thời gian frame ──
        fps_counter.end_frame()
        frame_idx += 1

        # In tiến độ mỗi 30 frames
        if frame_idx % 30 == 0:
            print(f"\r  Đang xử lý... Frame {frame_idx}/{info['total_frames']} | "
                  f"FPS: {fps_counter.get_fps():.1f}", end="", flush=True)

    # ── Bước 5: Dọn dẹp tài nguyên ──
    print(f"\n\n  Tổng frame đã xử lý: {frame_idx}")
    release_video(cap)
    if writer:
        writer.release()
        logger.info(f"Video đã lưu: {output_path}")
    if display:
        cv2.destroyAllWindows()

    logger.info("Hoàn thành!")
    return frame_idx


def demonstrate_image_concepts():
    """
    Demo các khái niệm cơ bản về ảnh số.
    Chạy phần này để hiểu numpy array trong context ảnh.
    """
    print("\n═══════════════════════════════════════")
    print("  DEMO: KHÁI NIỆM ẢNH SỐ")
    print("═══════════════════════════════════════\n")

    # ── 1. Ảnh là gì? ──
    print("1. Ảnh = Numpy Array 3 chiều (Height, Width, Channels)")
    # Tạo ảnh trắng 100x200 pixel
    white_img = np.ones((100, 200, 3), dtype=np.uint8) * 255
    print(f"   Ảnh trắng 200x100: shape={white_img.shape}")
    print(f"   dtype={white_img.dtype} | max={white_img.max()} | min={white_img.min()}")

    # ── 2. Màu trong OpenCV là BGR ──
    print("\n2. Màu sắc trong OpenCV = BGR (không phải RGB!)")
    red_pixel = np.array([[[0, 0, 255]]], dtype=np.uint8)    # Đỏ = R=255, G=0, B=0 → BGR=[0,0,255]
    green_pixel = np.array([[[0, 255, 0]]], dtype=np.uint8)  # Xanh lá
    blue_pixel = np.array([[[255, 0, 0]]], dtype=np.uint8)   # Xanh dương
    print(f"   Đỏ   (BGR): {red_pixel[0,0]}   → Nếu hiển thị bằng plt.imshow (RGB) sẽ thành xanh dương!")
    print(f"   Xanh lá    : {green_pixel[0,0]}")
    print(f"   Xanh dương : {blue_pixel[0,0]}")

    # ── 3. Truy cập pixel ──
    print("\n3. Truy cập và thay đổi pixel")
    img = np.zeros((200, 300, 3), dtype=np.uint8)  # Ảnh đen 300x200
    # Vẽ hình chữ nhật màu đỏ
    img[50:150, 100:200] = [0, 0, 255]  # [B, G, R] = [0, 0, 255] = Đỏ
    print(f"   Pixel tại (100, 150): {img[100, 150]}  ← Đỏ BGR=[0,0,255]")
    print(f"   Pixel tại (0, 0)    : {img[0, 0]}    ← Đen BGR=[0,0,0]")

    # ── 4. Lưu và đọc ảnh ──
    print("\n4. Lưu và đọc ảnh với OpenCV")
    os.makedirs("results", exist_ok=True)
    save_path = "results/demo_concepts.jpg"
    cv2.imwrite(save_path, img)
    loaded = cv2.imread(save_path)
    print(f"   Đã lưu: {save_path}")
    print(f"   Đã đọc lại: shape={loaded.shape}")

    # ── 5. Resize ảnh ──
    print("\n5. Resize ảnh")
    small = cv2.resize(img, (150, 100))  # (width, height) trong OpenCV
    print(f"   Gốc: {img.shape} → Resize → {small.shape}")

    print("\n  ✅ Hoàn thành demo khái niệm ảnh số!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 Demo - Video Basics")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Đường dẫn video hoặc '0' để dùng webcam"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Đường dẫn lưu video kết quả (tùy chọn)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Số frame tối đa để xử lý (tùy chọn, để test nhanh)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Không hiển thị cửa sổ (dùng khi chạy không có màn hình)"
    )
    parser.add_argument(
        "--concepts",
        action="store_true",
        help="Chạy demo khái niệm ảnh số (không cần video)"
    )

    args = parser.parse_args()

    if args.concepts or args.source is None:
        demonstrate_image_concepts()

    if args.source is not None:
        process_video(
            source=args.source,
            output_path=args.output,
            max_frames=args.max_frames,
            display=not args.no_display
        )
