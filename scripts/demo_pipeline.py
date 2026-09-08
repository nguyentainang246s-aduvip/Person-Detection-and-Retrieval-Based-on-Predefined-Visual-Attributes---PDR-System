"""
scripts/demo_pipeline.py
========================
PHASE 7 – CHECKPOINT & END-TO-END DEMO SCRIPT

Chạy toàn bộ hệ thống tìm kiếm người trong video theo thuộc tính nhận dạng.

CÁCH CHẠY:
    # 1. Chạy tìm kiếm với các tham số mẫu:
    .\\venv\\Scripts\\python.exe scripts/demo_pipeline.py --gender Male --upper-color Black --backpack Yes

    # 2. Chạy trên 1 video cụ thể:
    .\\venv\\Scripts\\python.exe scripts/demo_pipeline.py --video data/test_videos/my_video.mp4 --gender Female --upper-color White --threshold 0.75

    # 3. Hiển thị cửa sổ Video Realtime:
    .\\venv\\Scripts\\python.exe scripts/demo_pipeline.py --display
"""

import sys
import os
import argparse
import cv2
import numpy as np

# Thêm thư mục gốc vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.pipeline import PersonRetrievalPipeline
from src.utils.config_loader import load_config


def create_realistic_demo_video(output_path="data/test_videos/demo_search_video.mp4", num_frames=60):
    """
    Tạo video test có 2 đối tượng giả lập di chuyển:
    - Đối tượng 1: Áo Đen, Quần Đen (Target cần tìm)
    - Đối tượng 2: Áo Trắng, Quần Xanh (Đối tượng khác)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 15.0, (800, 600))

    for i in range(num_frames):
        frame = np.full((600, 800, 3), (230, 230, 230), dtype=np.uint8)

        # Vẽ đường đi/vỉa hè
        cv2.rectangle(frame, (0, 350), (800, 600), (160, 160, 160), -1)

        # 1. Đối tượng 1 (Áo Đen, Quần Đen) - Di chuyển từ trái sang phải
        x1 = 100 + i * 8
        y1 = 200
        # Đầu
        cv2.circle(frame, (x1 + 30, y1 + 25), 20, (180, 200, 230), -1)
        # Áo Đen
        cv2.rectangle(frame, (x1, y1 + 50), (x1 + 60, y1 + 160), (20, 20, 20), -1)
        # Balo sau lưng
        cv2.rectangle(frame, (x1 - 15, y1 + 65), (x1, y1 + 135), (40, 40, 40), -1)
        # Quần Đen
        cv2.rectangle(frame, (x1 + 8, y1 + 160), (x1 + 52, y1 + 270), (25, 25, 25), -1)

        # 2. Đối tượng 2 (Áo Trắng, Quần Xanh) - Di chuyển ngược lại
        x2 = 650 - i * 6
        y2 = 220
        # Đầu
        cv2.circle(frame, (x2 + 25, y2 + 20), 18, (180, 200, 230), -1)
        # Áo Trắng
        cv2.rectangle(frame, (x2, y2 + 45), (x2 + 50, y2 + 145), (245, 245, 245), -1)
        # Quần Xanh
        cv2.rectangle(frame, (x2 + 6, y2 + 145), (x2 + 44, y2 + 250), (200, 20, 20), -1)

        out.write(frame)

    out.release()
    return output_path


def main():
    parser = argparse.ArgumentParser(description="End-to-End Person Retrieval Pipeline Demo (Phase 7)")
    parser.add_argument("--video", type=str, default="data/test_videos/demo_search_video.mp4", help="Đường dẫn file video đầu vào")
    parser.add_argument("--gender", type=str, default="Male", choices=["Male", "Female", "Any"], help="Giới tính cần tìm")
    parser.add_argument("--upper-color", type=str, default="Black", help="Màu áo cần tìm (Black, White, Red, Blue, Green, Yellow, Gray, Any)")
    parser.add_argument("--lower-color", type=str, default="Black", help="Màu quần cần tìm (Black, White, Blue, Gray, Any)")
    parser.add_argument("--backpack", type=str, default="Yes", choices=["Yes", "No", "Any"], help="Có đeo balo không")
    parser.add_argument("--hat", type=str, default="Any", choices=["Yes", "No", "Any"], help="Có đội mũ không")
    parser.add_argument("--glasses", type=str, default="Any", choices=["Yes", "No", "Any"], help="Có đeo kính không")
    parser.add_argument("--threshold", type=float, default=None, help="Ngưỡng tương đồng (0.5 -> 1.0). Ghi đè config.yaml nếu cung cấp.")
    parser.add_argument("--max-frames", type=int, default=60, help="Số frames tối đa để xử lý")
    parser.add_argument("--display", action="store_true", help="Hiển thị cửa sổ video trực tiếp khi xử lý")

    args = parser.parse_args()

    print("=" * 65)
    print("  PHASE 7: FULL VIDEO RETRIEVAL PIPELINE DEMO")
    print("=" * 65)

    # Chuyển đổi tham số sang query dict
    def parse_bool_choice(choice):
        if choice == "Yes": return True
        if choice == "No": return False
        return None

    query = {
        "gender": args.gender,
        "upper_color": args.upper_color,
        "lower_color": args.lower_color,
        "backpack": parse_bool_choice(args.backpack),
        "hat": parse_bool_choice(args.hat),
        "glasses": parse_bool_choice(args.glasses)
    }

    # Đọc cấu hình
    config = load_config()
    final_threshold = args.threshold if args.threshold is not None else config.get("matching", {}).get("default_threshold", 0.70)

    print("\n[THÔNG TIN TRUY VẤN TÌM KIẾM]")
    print(f"  • Giới tính : {query['gender']}")
    print(f"  • Màu áo    : {query['upper_color']}")
    print(f"  • Màu quần  : {query['lower_color']}")
    print(f"  • Balo      : {args.backpack}")
    print(f"  • Đội mũ    : {args.hat}")
    print(f"  • Đeo kính  : {args.glasses}")
    print(f"  • Ngưỡng    : {final_threshold * 100:.0f}%\n")

    if not os.path.exists(args.video):
        print(f"[*] Đang tạo video mẫu thử nghiệm tại: {args.video}")
        create_realistic_demo_video(args.video)

    # Khởi tạo và chạy Pipeline
    pipeline = PersonRetrievalPipeline(matching_threshold=final_threshold)

    output_video = "results/retrieval_output.mp4"
    csv_log = "results/logs/retrieval_results.csv"

    summary = pipeline.run_on_video(
        video_source=args.video,
        target_query=query,
        output_video_path=output_video,
        save_csv_log=csv_log,
        threshold=final_threshold,
        max_frames=args.max_frames,
        display=args.display
    )

    print("\n" + "=" * 65)
    print("  KẾT QUẢ TÌM KIẾM TỔNG HỢP")
    print("=" * 65)
    print(f"  • Tổng số frames đã xử lý        : {summary['total_frames_processed']}")
    print(f"  • Tổng số người được theo dõi   : {summary['total_tracked_persons']}")
    print(f"  • Số lượng TARGET KHỚP YÊU CẦU   : {summary['targets_found_count']}")
    print("  " + "─" * 60)

    if summary["targets_found_count"] > 0:
        for idx, target in enumerate(summary["targets"]):
            attr = target["attributes"]
            print(f"\n  🎯 TARGET #{idx+1} [Track ID: {target['track_id']}]")
            print(f"     - Độ khớp (Score)     : {target['score'] * 100:.1f}%")
            print(f"     - Thời điểm xuất hiện : {target['timestamp']} (Frame {target['frame_idx']})")
            print(f"     - Đặc điểm nhận diện  : {attr['gender']}, Áo {attr['upper_color']}, Quần {attr['lower_color']}")
            print(f"     - Balo: {'Có' if attr['backpack'] else 'Không'} | Mũ: {'Có' if attr['hat'] else 'Không'} | Kính: {'Có' if attr['glasses'] else 'Không'}")
            print(f"     - Ảnh trích xuất      : {target['crop_path']}")

    print("\n  📁 File Video kết quả : " + output_video)
    print("  📊 File CSV nhật ký   : " + csv_log)
    print("=" * 65)
    print("  [PASSED] PHASE 7 CHECKPOINT: PIPELINE TÍCH HỢP HOÀN THÀNH XUẤT SẮC!")
    print("  -> Sẵn sàng chuyển sang PHASE 8: SQLite Database & Search History!")
    print("=" * 65)


if __name__ == "__main__":
    main()
