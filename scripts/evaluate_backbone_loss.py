"""
scripts/evaluate_backbone_loss.py
=================================
Thực nghiệm so sánh Backbone & Hàm mất mát (Loss Function) cho mô hình PAR (Giai đoạn 3).

Đo lường và so sánh:
  Bảng 1: So sánh Backbone
    - ResNet50 (Baseline)
    - MobileNetV3-Large (Khuyến nghị cho CPU/Edge)
    - EfficientNet-B0 (Tối ưu đa tỉ lệ)
    Chỉ số: Params (M), GFLOPs, mA (Val), Latency đo thật (ms/crop).

  Bảng 2: So sánh Loss Function
    - BCEWithLogitsLoss (Hiện tại)
    - Focal Loss (gamma=2.0)
    Chỉ số: mA tổng, F1 (Glasses - thuộc tính hiếm nhất), F1 (Hat).
"""

import os
import sys
import time
import json
import csv
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Fix UTF-8 stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.attributes.par_model import build_par_model, FocalLoss
from training.train_par import compute_par_metrics


def count_parameters(model: nn.Module) -> float:
    """Đếm tổng số tham số (triệu tham số - Millions)."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def measure_real_latency(model: nn.Module, device: str = "cpu", n_runs: int = 60) -> float:
    """Đo độ trễ suy luận thực tế (ms) trung bình qua n_runs."""
    model.eval()
    dev = torch.device(device)
    dummy_input = torch.randn(1, 3, 224, 112).to(dev)

    # Khởi động GPU/CPU (warmup)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    return float(np.mean(times))


def load_val_crops(max_crops_per_person: int = 12, train_ratio: float = 0.75):
    """
    Tải tập mẫu từ video 1 & 2 với phân chia ngắt kết nối theo Person ID (Disjoint Identity Split)
    nhằm loại bỏ 100% rò rỉ dữ liệu (data leakage) giữa các frame của cùng 1 người.
    """
    import random
    videos_info = [
        ("data/test_videos/4750042-hd_1920_1080_30fps.mp4", "data/gt/4750042_gt.csv"),
        ("data/test_videos/4750061-hd_1920_1080_30fps.mp4", "data/gt/4750061_gt.csv")
    ]

    crops_by_person = {}
    for vid_path, gt_csv in videos_info:
        if not os.path.exists(vid_path) or not os.path.exists(gt_csv):
            continue

        gt_by_frame = {}
        with open(gt_csv, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                gt_by_frame.setdefault(int(r["frame_idx"]), []).append(r)

        cap = cv2.VideoCapture(vid_path)
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in gt_by_frame:
                for r in gt_by_frame[frame_idx]:
                    pid = vid_path + "_" + r["person_id"]
                    if pid not in crops_by_person:
                        crops_by_person[pid] = []
                    if len(crops_by_person[pid]) < max_crops_per_person:
                        x = int(float(r["x"]))
                        y = int(float(r["y"]))
                        w = int(float(r["w"]))
                        h = int(float(r["h"]))
                        crop = frame[max(0, y):min(frame.shape[0], y+h), max(0, x):min(frame.shape[1], x+w)]
                        if crop.size > 0 and crop.shape[0] >= 30 and crop.shape[1] >= 15:
                            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            resized = cv2.resize(rgb, (112, 224))
                            tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
                            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                            tensor = (tensor - mean) / std

                            lbl = [
                                1.0 if r.get("gender") == "Female" else 0.0,
                                1.0 if int(r.get("hat", 0)) else 0.0,
                                1.0 if int(r.get("glasses", 0)) else 0.0,
                                1.0 if int(r.get("backpack", 0)) else 0.0
                            ]
                            crops_by_person[pid].append((tensor, lbl))
            frame_idx += 1
            if len(crops_by_person) >= 45 and all(len(v) >= 10 for v in crops_by_person.values()):
                break
        cap.release()

    if len(crops_by_person) == 0:
        raise FileNotFoundError(
            "Không tìm thấy data/gt/*.csv — hãy chạy extract_track_crops.py + "
            "label_attributes_app.py trước khi chạy ablation study."
        )

    pids = sorted(list(crops_by_person.keys()))
    random.seed(42)
    random.shuffle(pids)
    split = int(len(pids) * train_ratio)
    train_pids, eval_pids = pids[:split], pids[split:]

    train_data = [item for pid in train_pids for item in crops_by_person[pid]]
    eval_data = [item for pid in eval_pids for item in crops_by_person[pid]]

    train_x = torch.stack([x for x, y in train_data])
    train_y = torch.tensor([y for x, y in train_data], dtype=torch.float32)
    eval_x = torch.stack([x for x, y in eval_data])
    eval_y = torch.tensor([y for x, y in eval_data], dtype=torch.float32)

    return train_x, train_y, eval_x, eval_y, len(train_pids), len(eval_pids)


def run_stage3_experiments(output_dir: str = "results/benchmarks"):
    os.makedirs(output_dir, exist_ok=True)
    device = "cpu"
    print("=================================================================")
    print("📊 BẮT ĐẦU THỰC NGHIỆM GIAI ĐOẠN 3: SO SÁNH BACKBONE & LOSS FUNCTION")
    print("=================================================================")

    # 1. So sánh Backbone
    print("\n--- 1. ĐO ĐẠC & SO SÁNH CÁC BACKBONE ---")
    backbones_info = [
        ("ResNet50 (Hiện tại)", "resnet50", 4.10),
        ("MobileNetV3-Large", "mobilenet_v3", 0.23),
        ("EfficientNet-B0", "efficientnet_b0", 0.39),
    ]

    train_images, train_labels, eval_images, eval_labels, n_train_p, n_eval_p = load_val_crops()
    backbone_results = []

    print(f"[*] Phân chia Disjoint Identity: Train={n_train_p} người ({len(train_images)} crops) | Eval={n_eval_p} người ({len(eval_images)} crops)")

    for name, b_key, gflops in backbones_info:
        model = build_par_model(backbone=b_key, n_attrs=4, pretrained=True).to(device)
        params_m = count_parameters(model)
        latency_ms = measure_real_latency(model, device=device, n_runs=50)

        # Đo mA trên tập eval độc lập (những người chưa từng nhìn thấy)
        model.eval()
        with torch.no_grad():
            logits = model(eval_images)
            probs = torch.sigmoid(logits)
            metrics = compute_par_metrics(probs, eval_labels)

        ma_val = metrics["mA"] * 100.0
        backbone_results.append({
            "name": name,
            "backbone": b_key,
            "params_m": round(params_m, 2),
            "gflops": gflops,
            "ma": round(ma_val, 2),
            "latency_ms": round(latency_ms, 2)
        })
        print(f"  [+] {name:<22}: {params_m:5.2f}M params | {gflops:4.2f} GFLOPs | Latency: {latency_ms:5.2f} ms | mA: {ma_val:.2f}%")

    # 2. So sánh Loss Function (BCE vs Focal Loss)
    print("\n--- 2. SO SÁNH LOSS FUNCTION (HUẤN LUYỆN TRÊN TRAIN, ĐÁNH GIÁ TRÊN EVAL ĐỘC LẬP) ---")
    loss_experiments = []

    # Dataset huấn luyện chỉ chứa tập train
    dataset = TensorDataset(train_images, train_labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    losses = [
        ("BCE + Pos_weight (Baseline)", nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.2, 3.5, 5.0, 4.0]))),
        ("Focal Loss (gamma=2.0)", FocalLoss(gamma=2.0, alpha=0.75))
    ]

    for l_name, criterion in losses:
        # Huấn luyện thử nghiệm MobileNetV3 với từng hàm Loss trên tập train
        torch.manual_seed(42)
        m = build_par_model(backbone="mobilenet_v3", n_attrs=4, pretrained=True).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        m.train()
        for ep in range(6):
            for x_b, y_b in loader:
                opt.zero_grad()
                out = m(x_b)
                l = criterion(out, y_b)
                l.backward()
                opt.step()

        # Đánh giá nghiêm ngặt trên tập eval (những người chưa từng xuất hiện khi train)
        m.eval()
        with torch.no_grad():
            probs = torch.sigmoid(m(eval_images))
            met = compute_par_metrics(probs, eval_labels)

        # F1 kính (glasses - index 2), mũ (hat - index 1)
        f1_glasses = met["per_attr"][2]["f1"] * 100.0
        f1_hat = met["per_attr"][1]["f1"] * 100.0
        ma_total = met["mA"] * 100.0

        loss_experiments.append({
            "loss_name": l_name,
            "ma_total": round(ma_total, 2),
            "f1_glasses": round(f1_glasses, 2),
            "f1_hat": round(f1_hat, 2)
        })
        print(f"  [+] {l_name:<30}: mA tổng: {ma_total:.2f}% | F1 Glasses: {f1_glasses:.2f}% | F1 Hat: {f1_hat:.2f}%")

    # In Bảng 1: So sánh Backbone
    print("\n" + "=" * 75)
    print("📋 BẢNG 1: SO SÁNH HIỆU NĂNG BACKBONE (Ablation Study - Eval Set)")
    print("=" * 75)
    print(f"{'Backbone':<24} | {'Params':<8} | {'GFLOPs':<8} | {'mA (eval)':<10} | {'Latency đo thật'}")
    print("-" * 75)
    for r in backbone_results:
        print(f"{r['name']:<24} | {r['params_m']:5.2f}M   | {r['gflops']:5.2f}    | {r['ma']:6.2f}%    | {r['latency_ms']:5.2f} ms")
    print("-" * 75)

    # In Bảng 2: So sánh Loss
    print("\n" + "=" * 75)
    print("📋 BẢNG 2: SO SÁNH HÀM MẤT MÁT (KHÔNG RÒ RỈ DỮ LIỆU: EVAL SET ĐỘC LẬP)")
    print("=" * 75)
    print(f"{'Hàm Loss':<32} | {'mA Tổng':<10} | {'F1 (Glasses - hiếm)':<20} | {'F1 (Hat)':<10}")
    print("-" * 75)
    for r in loss_experiments:
        print(f"{r['loss_name']:<32} | {r['ma_total']:6.2f}%    | {r['f1_glasses']:6.2f}%              | {r['f1_hat']:6.2f}%")
    print("-" * 75 + "\n")

    # Lưu báo cáo Markdown và JSON
    md_path = os.path.join(output_dir, "par_ablation_study.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Báo Cáo Thực Nghiệm PAR: Backbone & Loss Function (Giai Đoạn 3)\n\n")
        f.write("### Bảng 1: So sánh Kiến trúc Backbone\n\n")
        f.write("| Backbone | Tham số (Params) | Khối lượng tính toán (GFLOPs) | Balanced mA (Eval) | Latency đo thật (CPU) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for r in backbone_results:
            f.write(f"| **{r['name']}** | {r['params_m']:.2f}M | {r['gflops']:.2f} GFLOPs | **{r['ma']:.2f}%** | **{r['latency_ms']:.2f} ms** |\n")

        f.write("\n### Bảng 2: So sánh Hàm mất mát (Loss Function — Đánh giá trên tập Eval độc lập)\n\n")
        f.write("| Hàm mất mát (Loss Function) | Balanced mA Tổng | F1 (Glasses — thuộc tính hiếm nhất) | F1 (Hat) |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for r in loss_experiments:
            f.write(f"| **{r['loss_name']}** | **{r['ma_total']:.2f}%** | **{r['f1_glasses']:.2f}%** | {r['f1_hat']:.2f}% |\n")

        # Ghi chú phương pháp luận
        f.write("\n> [!NOTE]\n")
        f.write(f"> **Phương pháp luận đánh giá & Kiểm soát Rò rỉ dữ liệu (Disjoint Identity / Data Leakage Control):**\n")
        f.write(f"> Thử nghiệm phân chia ngắt kết nối theo danh tính người (Disjoint Person ID Split): **{n_train_p} người** ({len(train_images)} crops) cho tập train và **{n_eval_p} người** ({len(eval_images)} crops) cho tập eval độc lập. Tất cả hình ảnh của cùng một người chỉ xuất hiện ở một trong hai tập, loại bỏ triệt để hiện tượng rò rỉ dữ liệu (data leakage) và học vẹt (overfitting) giữa các frame liên tiếp.\n")
        f.write(f"> \n")
        f.write(f"> **Lưu ý về quy mô dữ liệu:** Tập eval gồm {len(eval_images)} mẫu từ {n_eval_p} người chưa từng xuất hiện khi huấn luyện. Đây là kết quả thử nghiệm sơ bộ có đối chứng phương pháp luận chuẩn xác; để gia tăng tính khái quát hóa và độ tin cậy cho các thuộc tính hiếm (kính, mũ), cần tiếp tục mở rộng quy mô dữ liệu với các video còn lại hoặc kết hợp bộ benchmark quy mô lớn MSP60K.\n")

        # Tính toán động các chỉ số kết luận
        resnet = next(r for r in backbone_results if "resnet" in r["backbone"].lower())
        mobilenet = next(r for r in backbone_results if "mobilenet" in r["backbone"].lower())
        param_redux = (1.0 - mobilenet["params_m"] / resnet["params_m"]) * 100.0
        gflop_redux = (1.0 - mobilenet["gflops"] / resnet["gflops"]) * 100.0
        speedup = resnet["latency_ms"] / max(0.01, mobilenet["latency_ms"])

        bce_exp = next(r for r in loss_experiments if "BCE" in r["loss_name"])
        focal_exp = next(r for r in loss_experiments if "Focal" in r["loss_name"])
        delta_glasses = focal_exp["f1_glasses"] - bce_exp["f1_glasses"]
        delta_hat = focal_exp["f1_hat"] - bce_exp["f1_hat"]
        delta_ma = focal_exp["ma_total"] - bce_exp["ma_total"]

        f.write("\n### Kết luận nghiên cứu (Tính toán tự động từ số liệu thực nghiệm):\n")
        f.write(f"1. **MobileNetV3-Large** giảm **{param_redux:.1f}%** tham số ({mobilenet['params_m']:.2f}M vs {resnet['params_m']:.2f}M) và giảm **{gflop_redux:.1f}%** GFLOPs ({mobilenet['gflops']:.2f} vs {resnet['gflops']:.2f}) so với ResNet50, tốc độ suy luận nhanh hơn **{speedup:.2f}x** ({mobilenet['latency_ms']:.2f} ms vs {resnet['latency_ms']:.2f} ms).\n")
        f.write(f"2. **So sánh Hàm mất mát (Loss Function)**: Sau khi loại bỏ rò rỉ dữ liệu, Focal Loss đạt Balanced mA **{focal_exp['ma_total']:.2f}%** (so với BCE {bce_exp['ma_total']:.2f}%, chênh lệch {delta_ma:+.2f}%). Trên các thuộc tính hiếm, F1(Glasses) đạt **{focal_exp['f1_glasses']:.2f}%** (chênh lệch {delta_glasses:+.2f}%) và F1(Hat) đạt **{focal_exp['f1_hat']:.2f}%** (chênh lệch {delta_hat:+.2f}%).\n")

    json_path = os.path.join(output_dir, "par_ablation_study.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "data_split": {"train_size": len(train_images), "eval_size": len(eval_images)},
            "backbones": backbone_results,
            "loss_experiments": loss_experiments
        }, f, indent=2, ensure_ascii=False)

    print(f"[+] Đã xuất báo cáo Markdown: {md_path}")
    print(f"[+] Đã xuất dữ liệu JSON:     {json_path}")


if __name__ == "__main__":
    run_stage3_experiments()
