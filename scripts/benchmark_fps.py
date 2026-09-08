"""
scripts/benchmark_fps.py
========================
BENCHMARK HIỆU NĂNG HỆ THỐNG (FPS & LATENCY BENCHMARK)

Đo lường thời gian trễ (Latency tính bằng mili-giây ms) và tốc độ khung hình (FPS)
của từng module riêng lẻ cũng như toàn bộ Pipeline để phục vụ bảng số liệu Báo cáo ĐATN.

CÁCH CHẠY:
    .\\venv\\Scripts\\python.exe scripts/benchmark_fps.py
"""

import sys
import os
import time
import json
import datetime
import torch
import numpy as np
import cv2

# Thêm thư mục gốc vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.detector import PersonDetector
from src.tracking.tracker import PersonTracker
from src.attributes.color_detector import ColorDetector
from src.attributes.par_model import AttributeRecognizer
from src.retrieval.matcher import AttributeMatcher
from src.retrieval.pipeline import PersonRetrievalPipeline


def benchmark_module(name, func, num_iterations=50):
    """Đo thời gian chạy trung bình của một hàm qua num_iterations lần."""
    # Warm-up (chạy nháp 5 lần để nạp cache CPU/GPU)
    for _ in range(5):
        func()

    times = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0) # Đổi sang ms

    avg_ms = np.mean(times)
    std_ms = np.std(times)
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    return avg_ms, std_ms, fps


def run_all_benchmarks():
    print("=" * 65)
    print("  HỆ THỐNG ĐO LƯỜNG HIỆU NĂNG (BENCHMARK ĐỒ ÁN TỐT NGHIỆP)")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  • Phần cứng thực thi : {device.upper()}")
    if device == "cuda":
        print(f"  • GPU Model          : {torch.cuda.get_device_name(0)}")

    # Chuẩn bị dữ liệu mẫu
    frame_640x480 = np.full((480, 640, 3), (200, 200, 200), dtype=np.uint8)
    # Vẽ đối tượng giả lập
    cv2.circle(frame_640x480, (320, 150), 30, (50, 50, 200), -1)
    cv2.rectangle(frame_640x480, (280, 180), (360, 400), (20, 20, 20), -1)

    person_crop_224x112 = cv2.resize(frame_640x480[100:420, 260:380], (112, 224))

    # 1. Benchmark YOLO Detection
    detector = PersonDetector(device=device)
    t_det, s_det, fps_det = benchmark_module("YOLOv8n Detection", lambda: detector.detect(frame_640x480))

    # 2. Benchmark ByteTrack Tracking
    tracker = PersonTracker(device=device)
    t_trk, s_trk, fps_trk = benchmark_module("ByteTrack MOT", lambda: tracker.track(frame_640x480, persist=True))

    # 3. Benchmark Color Detector (HSV + KMeans)
    color_det = ColorDetector()
    t_col, s_col, fps_col = benchmark_module("HSV Color Detector", lambda: color_det.detect_colors(person_crop_224x112))

    # 4. Benchmark ResNet50 PAR Model
    par_model = AttributeRecognizer(device=device)
    t_par, s_par, fps_par = benchmark_module("ResNet50 PAR", lambda: par_model.predict(person_crop_224x112))

    # 5. Benchmark Matching Engine
    matcher = AttributeMatcher()
    sample_q = {"gender": "Male", "upper_color": "Black", "backpack": True}
    sample_attr = {"gender": "Male", "upper_color": "Black", "lower_color": "Blue", "backpack": True}
    t_mat, s_mat, fps_mat = benchmark_module("Matching Engine", lambda: matcher.is_match(sample_q, sample_attr))

    # 6. Benchmark Full Pipeline
    pipeline = PersonRetrievalPipeline(device=device)
    t_pip, s_pip, fps_pip = benchmark_module("Full Pipeline (End-to-End)", lambda: pipeline.process_frame(frame_640x480, 1, sample_q))

    # In Bảng Tổng Hợp Chuẩn Báo Cáo
    print("\n" + "=" * 65)
    print("  BẢNG KẾT QUẢ HIỆU NĂNG TỪNG MODULE (THỜI GIAN TRỄ & FPS)")
    print("=" * 65)
    print(f"{'Module / Thuật toán':<28} | {'Độ trễ trung bình':<18} | {'FPS ước tính':<12}")
    print("-" * 65)
    print(f"{'1. YOLOv8n Person Detection':<28} | {t_det:6.2f} ± {s_det:4.2f} ms   | {fps_det:6.1f} FPS")
    print(f"{'2. ByteTrack MOT Tracker':<28} | {t_trk:6.2f} ± {s_trk:4.2f} ms   | {fps_trk:6.1f} FPS")
    print(f"{'3. HSV Color Detector':<28} | {t_col:6.2f} ± {s_col:4.2f} ms   | {fps_col:6.1f} FPS")
    print(f"{'4. ResNet50 PAR Classifier':<28} | {t_par:6.2f} ± {s_par:4.2f} ms   | {fps_par:6.1f} FPS")
    print(f"{'5. Attribute Matching Engine':<28} | {t_mat:6.4f} ± {s_mat:4.4f} ms | {fps_mat:8.1f} FPS")
    print("-" * 65)
    print(f"{'6. FULL PIPELINE TÍCH HỢP':<28} | {t_pip:6.2f} ± {s_pip:4.2f} ms   | {fps_pip:6.1f} FPS")
    print("=" * 65)

    print("\n💡 GHI CHÚ BÁO CÁO:")
    print("  • Bảng số liệu trên có thể copy trực tiếp vào Chương 4 (Thực nghiệm & Đánh giá) của ĐATN.")
    print("  • Với cơ chế Caching thuộc tính (chỉ chạy ResNet mỗi 5 frames), Full Pipeline duy trì tốc độ realtime mượt mà.")

    # P0-1: Lưu kết quả ra JSON
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "benchmark_results.json")
    
    device_name = "cuda (" + torch.cuda.get_device_name(0) + ")" if device == "cuda" else "cpu"
    
    benchmark_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "device": device_name,
        "modules": [
            {"name": "YOLOv8 Detection", "latency_ms": round(t_det, 2), "std_ms": round(s_det, 2), "fps": round(fps_det, 1)},
            {"name": "ByteTrack MOT", "latency_ms": round(t_trk, 2), "std_ms": round(s_trk, 2), "fps": round(fps_trk, 1)},
            {"name": "K-Means Color HSV", "latency_ms": round(t_col, 2), "std_ms": round(s_col, 2), "fps": round(fps_col, 1)},
            {"name": "ResNet50 PAR", "latency_ms": round(t_par, 2), "std_ms": round(s_par, 2), "fps": round(fps_par, 1)},
            {"name": "Attribute Matching Engine", "latency_ms": round(t_mat, 4), "std_ms": round(s_mat, 4), "fps": round(fps_mat, 1)},
            {"name": "Overall Pipeline", "latency_ms": round(t_pip, 2), "std_ms": round(s_pip, 2), "fps": round(fps_pip, 1)}
        ]
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(benchmark_data, f, ensure_ascii=False, indent=4)
        
    print(f"\n[INFO] Đã lưu kết quả benchmark ra file: {json_path}")


if __name__ == "__main__":
    run_all_benchmarks()
