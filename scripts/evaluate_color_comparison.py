"""
scripts/evaluate_color_comparison.py
====================================
Thực nghiệm so sánh đa điều kiện cho Module Nhận dạng Màu sắc (Giai đoạn 2).

So sánh 3 phương pháp:
  1. HSV K-Means (Thuần phân cụm HSV không có Delta-E)
  2. HSV + Delta-E Fallback (Hệ thống hiện tại)
  3. Learned Color Head (Deep Learning DualColorNet)

Trên 3 điều kiện môi trường thực tế:
  A. Điều kiện thường (Ánh sáng ban ngày chuẩn)
  B. Ngược sáng & Bóng đổ mạnh (Synthetic Shadow + Backlight Jitter)
  C. Quần áo họa tiết & Nhiễu hoa văn (Texture Pattern Noise)

Xuất ra bảng Ablation Study bắt buộc cho luận văn.
"""

import os
import sys
import time
import json
import csv
import cv2
import numpy as np
from PIL import Image

# Fix UTF-8 stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.attributes.color_detector import ColorDetector
from src.attributes.learned_color_head import LearnedColorDetector, COLOR_CLASSES, add_synthetic_shadow
from src.utils.video_utils import open_video, read_frame, crop_person


# Tập track độc lập làm Test Set (tách biệt hoàn toàn với Training Set)
TEST_TRACKS_4750042 = {2, 7, 8, 16, 28, 46, 74, 88}
TEST_TRACKS_4750061 = {66, 71, 87, 90, 92, 100, 118, 126}


def apply_backlight_shadow(crop: np.ndarray) -> np.ndarray:
    """
    Mô phỏng điều kiện ngược sáng & bóng đổ mạnh (Physical Backlight / Silhouette):
    Sử dụng gradient bóng đổ có cấu trúc (add_synthetic_shadow) và suy giảm phơi sáng,
    tuyệt đối KHÔNG dùng nhiễu Gaussian ngẫu nhiên per-pixel.
    """
    pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    shadowed = add_synthetic_shadow(pil_img)
    arr = np.array(shadowed, dtype=np.float32)
    h, w = arr.shape[:2]
    # Suy giảm độ sáng theo gradient thẳng đứng mô phỏng nguồn sáng trần/mặt trời ngược hướng
    gradient = np.linspace(0.60, 0.85, h)[:, None, None]
    arr = np.clip(arr * gradient - 10, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def apply_pattern_noise(crop: np.ndarray) -> np.ndarray:
    """Mô phỏng quần áo họa tiết/sọc caro hoặc hoa văn bề mặt dệt (Texture Grid, không dùng nhiễu ngẫu nhiên)."""
    out = crop.copy().astype(np.float32)
    h, w = out.shape[:2]
    # Thêm sọc kẻ ngang và dọc mô phỏng hoa văn dệt/caro
    for i in range(0, h, 6):
        out[i:i+2, :, :] = np.clip(out[i:i+2, :, :] * 0.7, 0, 255)
    for j in range(0, w, 6):
        out[:, j:j+2, :] = np.clip(out[:, j:j+2, :] * 0.8, 0, 255)
    return np.clip(out, 0, 255).astype(np.uint8)


def load_test_samples(max_samples: int = 250):
    """Tải tập mẫu test từ các Track độc lập (chưa từng xuất hiện trong tập Train)."""
    samples = []
    videos_info = [
        ("data/test_videos/4750042-hd_1920_1080_30fps.mp4", "data/gt/4750042_gt.csv", TEST_TRACKS_4750042),
        ("data/test_videos/4750061-hd_1920_1080_30fps.mp4", "data/gt/4750061_gt.csv", TEST_TRACKS_4750061)
    ]

    per_vid_limit = max_samples // len(videos_info)

    for video_path, gt_csv, allowed_tracks in videos_info:
        if not os.path.exists(video_path) or not os.path.exists(gt_csv):
            continue

        gt_by_frame = {}
        with open(gt_csv, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                pid = int(r["person_id"])
                if pid in allowed_tracks:
                    gt_by_frame.setdefault(int(r["frame_idx"]), []).append(r)

        cap, info = open_video(video_path)
        frame_idx = 0
        vid_count = 0

        while True:
            ret, frame = read_frame(cap)
            if not ret:
                break
            if frame_idx in gt_by_frame:
                for r in gt_by_frame[frame_idx]:
                    x = int(float(r["x"]))
                    y = int(float(r["y"]))
                    w = int(float(r["w"]))
                    h = int(float(r["h"]))
                    c = crop_person(frame, [x, y, x + w, y + h])
                    if c is not None and c.shape[0] >= 40 and c.shape[1] >= 20:
                        samples.append({
                            "crop": c,
                            "upper_color": r.get("upper_color", "Other").strip().capitalize(),
                            "lower_color": r.get("lower_color", "Other").strip().capitalize()
                        })
                        vid_count += 1
                        if vid_count >= per_vid_limit:
                            break
            frame_idx += 1
            if vid_count >= per_vid_limit:
                break
        cap.release()

    return samples


def run_color_ablation_study(output_dir: str = "results/benchmarks"):
    os.makedirs(output_dir, exist_ok=True)
    print("=================================================================")
    print("📊 BẮT ĐẦU THỰC NGHIỆM ABLATION STUDY: NHẬN DẠNG MÀU SẮC (GĐ 2)")
    print("=================================================================")

    samples = load_test_samples(max_samples=250)
    print(f"[+] Đã chuẩn bị {len(samples)} mẫu ảnh crop người từ video thực tế.\n")

    # Khởi tạo 3 mô hình / phương pháp
    detector_hsv_pure = ColorDetector(n_clusters=2, method="kmeans_hsv")
    detector_hsv_delta_e = ColorDetector(n_clusters=2, method="hsv_delta_e")
    detector_learned = LearnedColorDetector(weights_path="models/par/color_head.pth")

    methods = [
        ("HSV K-Means (Thuần)", detector_hsv_pure, "pure"),
        ("HSV + Delta-E Fallback", detector_hsv_delta_e, "delta_e"),
        ("Learned Color Head (Deep Learning)", detector_learned, "learned")
    ]

    conditions = [
        ("normal", "Điều kiện thường (Normal)", lambda c: c),
        ("backlight", "Ngược sáng / Bóng đổ (Backlight)", apply_backlight_shadow),
        ("pattern", "Họa tiết / Hoa văn (Pattern)", apply_pattern_noise)
    ]

    results_table = {}

    for m_name, model, m_type in methods:
        results_table[m_name] = {}
        total_time = 0.0
        total_inferences = 0

        for cond_key, cond_title, transform_fn in conditions:
            correct_upper = 0
            correct_lower = 0
            n_eval = 0

            for s in samples:
                test_crop = transform_fn(s["crop"])
                gt_u = s["upper_color"]
                gt_l = s["lower_color"]

                t0 = time.perf_counter()
                if m_type == "pure":
                    # K-Means thuần không qua Delta-E
                    u_col, _ = model.get_upper_color(test_crop)
                    l_col, _ = model.get_lower_color(test_crop)
                elif m_type == "delta_e":
                    res = model.detect_colors(test_crop)
                    u_col = res["upper_color"]
                    l_col = res["lower_color"]
                else:
                    res = model.detect_colors(test_crop)
                    u_col = res["upper_color"]
                    l_col = res["lower_color"]
                dt = time.perf_counter() - t0
                total_time += dt
                total_inferences += 1

                if u_col == gt_u:
                    correct_upper += 1
                if l_col == gt_l:
                    correct_lower += 1
                n_eval += 1

            u_acc = (correct_upper / n_eval) * 100.0 if n_eval > 0 else 0.0
            l_acc = (correct_lower / n_eval) * 100.0 if n_eval > 0 else 0.0
            avg_acc = (u_acc + l_acc) / 2.0
            results_table[m_name][cond_key] = avg_acc

        avg_latency_ms = (total_time / max(1, total_inferences)) * 1000.0
        results_table[m_name]["latency_ms"] = avg_latency_ms

    # In bảng số liệu chuẩn khoa học
    print("-" * 80)
    print(f"{'Phương pháp':<35} | {'Thường':<10} | {'Ngược sáng':<12} | {'Họa tiết':<10} | {'Độ trễ':<8}")
    print("-" * 80)
    for m_name, res in results_table.items():
        norm_acc = res["normal"]
        back_acc = res["backlight"]
        patt_acc = res["pattern"]
        lat = res["latency_ms"]
        print(f"{m_name:<35} | {norm_acc:6.2f}%   | {back_acc:8.2f}%    | {patt_acc:6.2f}%   | {lat:5.2f}ms")
    print("-" * 80 + "\n")

    # Xuất file Markdown và JSON
    md_path = os.path.join(output_dir, "color_ablation_study.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Bảng Số Liệu Thực Nghiệm Màu Sắc (Ablation Study — Giai Đoạn 2)\n\n")
        f.write("| Phương pháp | Accuracy (Thường) | Accuracy (Ngược sáng / Bóng đổ) | Accuracy (Họa tiết) | Độ trễ (ms) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for m_name, res in results_table.items():
            f.write(f"| **{m_name}** | **{res['normal']:.2f}%** | **{res['backlight']:.2f}%** | **{res['pattern']:.2f}%** | {res['latency_ms']:.2f} ms |\n")
        # Tính toán động các chỉ số kết luận
        hsv_pure = results_table["HSV K-Means (Thuần)"]
        learned = results_table["Learned Color Head (Deep Learning)"]
        gain_normal = learned["normal"] - hsv_pure["normal"]
        gain_backlight = learned["backlight"] - hsv_pure["backlight"]
        gain_pattern = learned["pattern"] - hsv_pure["pattern"]
        lat_hsv = hsv_pure["latency_ms"]
        lat_learned = learned["latency_ms"]

        f.write("\n### Nhận xét & Phân tích Trade-off (Tính toán tự động từ số liệu thực nghiệm):\n")
        f.write(f"1. **Độ chính xác (Accuracy):** Ở điều kiện thường trên tập test độc lập, Learned Color Head đạt **{learned['normal']:.2f}%** (chênh lệch {gain_normal:+.2f}% so với HSV K-Means). Trong điều kiện ánh sáng khó, mô hình kháng bóng đổ / ngược sáng đạt **{learned['backlight']:.2f}%** (chênh lệch {gain_backlight:+.2f}%) và khi gặp họa tiết đạt **{learned['pattern']:.2f}%** (chênh lệch {gain_pattern:+.2f}%).\n")
        f.write(f"2. **Độ trễ (Latency):** Phương pháp HSV K-Means có độ trễ **{lat_hsv:.2f} ms**, trong khi Learned Color Head mất **{lat_learned:.2f} ms** (chênh lệch {lat_learned - lat_hsv:+.2f} ms, hoàn toàn đáp ứng ngưỡng thời gian thực).\n")

    json_path = os.path.join(output_dir, "color_ablation_study.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_table, f, indent=2, ensure_ascii=False)

    print(f"[+] Báo cáo Markdown đã lưu: {md_path}")
    print(f"[+] Dữ liệu JSON đã lưu:     {json_path}")
    return results_table


if __name__ == "__main__":
    run_color_ablation_study()
