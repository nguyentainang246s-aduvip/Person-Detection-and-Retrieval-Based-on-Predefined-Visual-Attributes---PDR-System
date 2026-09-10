"""
Đánh giá chuyên sâu trên Video Test (Video Benchmark Evaluation)
Thuộc Giai đoạn 0 của lộ trình nâng cấp PDR-System.

Đo lường tự động:
  1. Tracking Metrics (chuẩn motmetrics):
     - MOTA (Multiple Object Tracking Accuracy)
     - MOTP (Multiple Object Tracking Precision)
     - IDF1 (Identification F1 Score)
     - IDP, IDR (ID Precision & Recall)
     - IDSW (ID Switch Count)
     - FP, FN (False Positives, Misses)
  2. PAR Attribute Metrics:
     - mA (Mean Balanced Accuracy) chuẩn PAR literature
     - Precision, Recall, F1 cho từng thuộc tính (gender, hat, glasses, backpack)
  3. Color Recognition Metrics:
     - Upper Color Accuracy & Lower Color Accuracy
  4. System Performance:
     - Throughput (FPS) & Latency per frame (ms)
"""

import os
import sys
import argparse
import csv
import json
import time
import numpy as np
from collections import defaultdict
import motmetrics as mm

# Thêm thư mục gốc vào PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.retrieval.pipeline import PersonRetrievalPipeline
from src.utils.video_utils import open_video, read_frame


def compute_iou(box1, box2):
    """
    Tính IoU giữa 2 box [x, y, w, h].
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter_area = max(0, xb - xa) * max(0, yb - ya)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def load_ground_truth(gt_csv_path: str):
    """
    Đọc file Ground-Truth CSV.
    Cấu trúc mong đợi:
      frame_idx, person_id, x, y, w, h, gender, upper_color, lower_color, hat, glasses, backpack
    """
    gt_by_frame = defaultdict(list)
    if not os.path.exists(gt_csv_path):
        raise FileNotFoundError(f"Không tìm thấy file Ground Truth: {gt_csv_path}")

    with open(gt_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            f_idx = int(row["frame_idx"])
            gt_by_frame[f_idx].append({
                "person_id": int(row["person_id"]),
                "box": [float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"])],
                "gender": row.get("gender", "").strip().capitalize(),
                "upper_color": row.get("upper_color", "").strip().capitalize(),
                "lower_color": row.get("lower_color", "").strip().capitalize(),
                "hat": int(row.get("hat", 0)) in [1, "1", True, "true", "True"],
                "glasses": int(row.get("glasses", 0)) in [1, "1", True, "true", "True"],
                "backpack": int(row.get("backpack", 0)) in [1, "1", True, "true", "True"]
            })
    return gt_by_frame


def evaluate_video(
    video_path: str,
    gt_path: str,
    output_dir: str = "results/benchmarks",
    iou_threshold: float = 0.5,
    tracker_name: str = "bytetrack"
):
    """Thực hiện đánh giá toàn diện video theo Ground Truth."""
    os.makedirs(output_dir, exist_ok=True)
    gt_by_frame = load_ground_truth(gt_path)

    print(f"\n========================================================")
    print(f"[*] BẮT ĐẦU ĐÁNH GIÁ THỰC NGHIỆM TRÊN VIDEO")
    print(f"[*] Video:        {video_path}")
    print(f"[*] Ground Truth: {gt_path}")
    print(f"[*] Tracker:      {tracker_name}")
    print(f"========================================================")

    # Khởi tạo Pipeline theo tracker yêu cầu
    if tracker_name == "bytetrack_baseline":
        pipeline = PersonRetrievalPipeline(tracker_type="bytetrack.yaml", enable_reid=False)
    elif tracker_name in ["botsort", "botsort_reid"]:
        pipeline = PersonRetrievalPipeline(tracker_type="config/botsort_reid.yaml", enable_reid=True)
    elif tracker_name == "bytetrack_reid":
        pipeline = PersonRetrievalPipeline(tracker_type="bytetrack.yaml", enable_reid=True)
    else:
        pipeline = PersonRetrievalPipeline(tracker_type=tracker_name)

    cap, info = open_video(video_path)
    total_frames = info["total_frames"]
    fps_video = info["fps"] or 30.0

    # Khởi tạo MOT Accumulator
    acc = mm.MOTAccumulator(auto_id=True)

    # Bộ chứa dữ liệu đánh giá thuộc tính (ground_truth vs predicted)
    attr_eval = {
        "gender": {"y_true": [], "y_pred": []},
        "hat": {"y_true": [], "y_pred": []},
        "glasses": {"y_true": [], "y_pred": []},
        "backpack": {"y_true": [], "y_pred": []},
        "upper_color": {"y_true": [], "y_pred": []},
        "lower_color": {"y_true": [], "y_pred": []},
    }

    frame_times = []
    frame_idx = 0

    while True:
        t_start = time.perf_counter()
        ret, frame = read_frame(cap)
        if not ret:
            break

        # Chạy pipeline phân tích frame hiện tại
        _, targets = pipeline.process_frame(frame, frame_idx, target_query={})
        t_elapsed = time.perf_counter() - t_start
        frame_times.append(t_elapsed)

        # Lấy Ground Truth của frame hiện tại
        gt_objs = gt_by_frame.get(frame_idx, [])
        gt_ids = [obj["person_id"] for obj in gt_objs]
        gt_boxes = [obj["box"] for obj in gt_objs]

        # Lấy dự đoán trực tiếp từ frame vừa xử lý (không gọi lại tracker lần 2)
        pred_objs = getattr(pipeline, "last_tracked_objects", [])
        pred_ids = []
        pred_boxes = []
        pred_attrs = {}

        for p in pred_objs:
            tid = p["track_id"]
            box = p["bbox"] # [x1, y1, x2, y2]
            w = max(1, box[2] - box[0])
            h = max(1, box[3] - box[1])
            pred_ids.append(tid)
            pred_boxes.append([box[0], box[1], w, h])
            # Thuộc tính lấy từ track_memory
            mem = pipeline.track_memory.get(tid, {})
            pred_attrs[tid] = mem.get("attributes", {})

        # Tính ma trận khoảng cách IoU (1 - IoU) giữa GT và Pred
        num_gt = len(gt_objs)
        num_pred = len(pred_objs)
        dist_matrix = np.full((num_gt, num_pred), np.nan)

        for i, gt_box in enumerate(gt_boxes):
            for j, p_box in enumerate(pred_boxes):
                iou = compute_iou(gt_box, p_box)
                if iou >= iou_threshold:
                    dist_matrix[i, j] = 1.0 - iou # khoảng cách càng nhỏ càng khớp

        # Cập nhật vào MOT Accumulator
        acc.update(gt_ids, pred_ids, dist_matrix)

        # Thu thập cặp đã ghép đúng để đánh giá thuộc tính
        for i, gt_obj in enumerate(gt_objs):
            # Tìm pred có IoU cao nhất
            best_iou = 0.0
            best_p_idx = -1
            for j, p_box in enumerate(pred_boxes):
                iou = compute_iou(gt_obj["box"], p_box)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_p_idx = j

            if best_p_idx >= 0:
                p_id = pred_ids[best_p_idx]
                p_attr = pred_attrs.get(p_id, {})

                if "gender" in p_attr and gt_obj["gender"]:
                    attr_eval["gender"]["y_true"].append(gt_obj["gender"])
                    attr_eval["gender"]["y_pred"].append(p_attr["gender"])

                if "hat" in p_attr:
                    attr_eval["hat"]["y_true"].append(int(gt_obj["hat"]))
                    attr_eval["hat"]["y_pred"].append(1 if p_attr.get("hat") else 0)

                if "glasses" in p_attr:
                    attr_eval["glasses"]["y_true"].append(int(gt_obj["glasses"]))
                    attr_eval["glasses"]["y_pred"].append(1 if p_attr.get("glasses") else 0)

                if "backpack" in p_attr:
                    attr_eval["backpack"]["y_true"].append(int(gt_obj["backpack"]))
                    attr_eval["backpack"]["y_pred"].append(1 if p_attr.get("backpack") else 0)

                if "upper_color" in p_attr and gt_obj["upper_color"]:
                    attr_eval["upper_color"]["y_true"].append(gt_obj["upper_color"])
                    attr_eval["upper_color"]["y_pred"].append(p_attr["upper_color"])

                if "lower_color" in p_attr and gt_obj["lower_color"]:
                    attr_eval["lower_color"]["y_true"].append(gt_obj["lower_color"])
                    attr_eval["lower_color"]["y_pred"].append(p_attr["lower_color"])

        frame_idx += 1
        if frame_idx % 50 == 0 or frame_idx == total_frames:
            print(f"  [>] Đang xử lý: {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%)")

    cap.release()

    # =========================================================================
    # 1. TÍNH TOÁN METRICS TRACKING (motmetrics)
    # =========================================================================
    mh = mm.metrics.create()
    mot_summary = mh.compute(
        acc,
        metrics=[
            "mota", "motp", "idf1", "idp", "idr",
            "num_switches", "num_false_positives", "num_misses",
            "mostly_tracked", "mostly_lost"
        ],
        name=tracker_name
    )

    mota = float(mot_summary.get("mota", [0]).iloc[0]) * 100.0
    idf1 = float(mot_summary.get("idf1", [0]).iloc[0]) * 100.0
    idp = float(mot_summary.get("idp", [0]).iloc[0]) * 100.0
    idr = float(mot_summary.get("idr", [0]).iloc[0]) * 100.0
    idsw = int(mot_summary.get("num_switches", [0]).iloc[0])
    fp = int(mot_summary.get("num_false_positives", [0]).iloc[0])
    fn = int(mot_summary.get("num_misses", [0]).iloc[0])

    # =========================================================================
    # 2. TÍNH TOÁN METRICS THUỘC TÍNH (PAR & COLOR)
    # =========================================================================
    def calc_binary_metrics(y_true, y_pred):
        if not y_true:
            return {"acc": 0.0, "prec": 0.0, "rec": 0.0, "f1": 0.0, "ma": 0.0}
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        p_pos = max(1, tp + fn)
        p_neg = max(1, tn + fp)

        acc = (tp + tn) / max(1, len(y_true))
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = (2 * prec * rec) / max(1e-6, prec + rec) if (prec + rec) > 0 else 0.0
        # mA chuẩn PAR
        ma = 0.5 * (tp / p_pos + tn / p_neg)

        return {
            "acc": float(acc) * 100.0,
            "prec": float(prec) * 100.0,
            "rec": float(rec) * 100.0,
            "f1": float(f1) * 100.0,
            "ma": float(ma) * 100.0,
            "count": len(y_true)
        }

    def calc_multiclass_acc(y_true, y_pred):
        if not y_true:
            return 0.0, 0
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        return float(correct / len(y_true)) * 100.0, len(y_true)

    # Đổi gender về nhị phân (1: Male, 0: Female)
    g_true = [1 if g == "Male" else 0 for g in attr_eval["gender"]["y_true"]]
    g_pred = [1 if g == "Male" else 0 for g in attr_eval["gender"]["y_pred"]]

    gender_res = calc_binary_metrics(g_true, g_pred)
    hat_res = calc_binary_metrics(attr_eval["hat"]["y_true"], attr_eval["hat"]["y_pred"])
    gla_res = calc_binary_metrics(attr_eval["glasses"]["y_true"], attr_eval["glasses"]["y_pred"])
    bag_res = calc_binary_metrics(attr_eval["backpack"]["y_true"], attr_eval["backpack"]["y_pred"])

    # PAR Mean Accuracy (mA)
    par_ma_list = [gender_res["ma"], hat_res["ma"], gla_res["ma"], bag_res["ma"]]
    par_mean_ma = float(np.mean(par_ma_list))

    # Color Metrics
    upper_acc, upper_cnt = calc_multiclass_acc(attr_eval["upper_color"]["y_true"], attr_eval["upper_color"]["y_pred"])
    lower_acc, lower_cnt = calc_multiclass_acc(attr_eval["lower_color"]["y_true"], attr_eval["lower_color"]["y_pred"])
    avg_color_acc = (upper_acc + lower_acc) / 2.0

    # Tốc độ
    avg_latency_ms = float(np.mean(frame_times)) * 1000.0 if frame_times else 0.0
    throughput_fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

    # =========================================================================
    # 3. HIỂN THỊ KẾT QUẢ DẠNG BẢNG ĐẸP MẮT
    # =========================================================================
    print("\n" + "="*65)
    print(f"📊 BÁO CÁO ĐÁNH GIÁ THỰC NGHIỆM TOÀN DIỆN (Ablation Benchmark)")
    print("="*65)
    print(f"1. KẾT QUẢ THEO DÕI ĐỐI TƯỢNG (TRACKING METRICS - motmetrics):")
    print(f"   - MOTA (Tracking Accuracy):      {mota:6.2f} %")
    print(f"   - IDF1 (ID Preservation F1):     {idf1:6.2f} % (Trọng tâm Re-ID)")
    print(f"   - IDP / IDR (Precision / Recall):{idp:6.2f}% / {idr:6.2f}%")
    print(f"   - ID-Switches (Số lần đổi ID):   {idsw:6d} lần")
    print(f"   - False Positives (Báo ảo):      {fp:6d}")
    print(f"   - Misses (Bỏ sót):               {fn:6d}")
    print("-"*65)
    print(f"2. KẾT QUẢ NHẬN DẠNG THUỘC TÍNH (PAR METRICS):")
    print(f"   - mA Trung bình (Mean Balanced): {par_mean_ma:6.2f} %")
    print(f"   + Giới tính (Gender):            mA: {gender_res['ma']:5.1f}% | F1: {gender_res['f1']:5.1f}% | Acc: {gender_res['acc']:5.1f}%")
    print(f"   + Đội mũ (Hat):                  mA: {hat_res['ma']:5.1f}% | F1: {hat_res['f1']:5.1f}% | Acc: {hat_res['acc']:5.1f}%")
    print(f"   + Đeo kính (Glasses):            mA: {gla_res['ma']:5.1f}% | F1: {gla_res['f1']:5.1f}% | Acc: {gla_res['acc']:5.1f}%")
    print(f"   + Balo/Túi (Backpack):           mA: {bag_res['ma']:5.1f}% | F1: {bag_res['f1']:5.1f}% | Acc: {bag_res['acc']:5.1f}%")
    print("-"*65)
    print(f"3. KẾT QUẢ NHẬN DẠNG MÀU SẮC (COLOR METRICS):")
    print(f"   - Màu áo (Upper Color Accuracy): {upper_acc:6.2f} % ({upper_cnt} mẫu)")
    print(f"   - Màu quần (Lower Color Acc):    {lower_acc:6.2f} % ({lower_cnt} mẫu)")
    print(f"   - Độ chính xác màu trung bình:   {avg_color_acc:6.2f} %")
    print("-"*65)
    print(f"4. HIỆU NĂNG TỐC ĐỘ (SPEED & LATENCY):")
    print(f"   - Độ trễ trung bình / frame:     {avg_latency_ms:6.1f} ms")
    print(f"   - Tốc độ xử lý thực tế:          {throughput_fps:6.1f} FPS")
    print("="*65 + "\n")

    # =========================================================================
    # 4. LƯU BÁO CÁO JSON VÀ MARKDOWN
    # =========================================================================
    results = {
        "video_path": video_path,
        "gt_path": gt_path,
        "tracker": tracker_name,
        "total_frames": frame_idx,
        "tracking": {
            "mota": round(mota, 2),
            "idf1": round(idf1, 2),
            "idp": round(idp, 2),
            "idr": round(idr, 2),
            "id_switches": idsw,
            "false_positives": fp,
            "misses": fn
        },
        "par": {
            "mA": round(par_mean_ma, 2),
            "gender": gender_res,
            "hat": hat_res,
            "glasses": gla_res,
            "backpack": bag_res
        },
        "color": {
            "upper_color_accuracy": round(upper_acc, 2),
            "lower_color_accuracy": round(lower_acc, 2),
            "average_color_accuracy": round(avg_color_acc, 2)
        },
        "performance": {
            "latency_ms": round(avg_latency_ms, 1),
            "fps": round(throughput_fps, 1)
        }
    }

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    json_path = os.path.join(output_dir, f"benchmark_{base_name}_{tracker_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[+] Đã xuất kết quả JSON: {json_path}")

    md_path = os.path.join(output_dir, f"benchmark_{base_name}_{tracker_name}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Kết Quả Thực Nghiệm — {base_name}\n\n")
        f.write(f"- **Video:** `{video_path}` ({frame_idx} frames)\n")
        f.write(f"- **Tracker:** `{tracker_name}`\n")
        f.write(f"- **Tốc độ:** {throughput_fps:.1f} FPS ({avg_latency_ms:.1f} ms/frame)\n\n")
        f.write("### 1. Chỉ số Tracking (motmetrics)\n\n")
        f.write("| Chỉ số | Giá trị | Ý nghĩa |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write(f"| **MOTA** | **{mota:.2f}%** | Độ chính xác tracking tổng quát |\n")
        f.write(f"| **IDF1** | **{idf1:.2f}%** | Khả năng duy trì đúng ID (Trọng tâm Re-ID) |\n")
        f.write(f"| **ID-Switches** | **{idsw} lần** | Số lần bị nhảy nhầm ID |\n")
        f.write(f"| **False Positives** | {fp} | Số phát hiện nhầm (báo ảo) |\n")
        f.write(f"| **Misses** | {fn} | Số đối tượng bị bỏ sót |\n\n")
        f.write("### 2. Chỉ số Nhận dạng Thuộc tính (PAR)\n\n")
        f.write(f"- **Mean Accuracy (mA chuẩn PAR):** **{par_mean_ma:.2f}%**\n\n")
        f.write("| Thuộc tính | mA (%) | Precision (%) | Recall (%) | F1 Score (%) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        f.write(f"| **Giới tính (Gender)** | {gender_res['ma']:.1f} | {gender_res['prec']:.1f} | {gender_res['rec']:.1f} | {gender_res['f1']:.1f} |\n")
        f.write(f"| **Đội mũ (Hat)** | {hat_res['ma']:.1f} | {hat_res['prec']:.1f} | {hat_res['rec']:.1f} | {hat_res['f1']:.1f} |\n")
        f.write(f"| **Kính mắt (Glasses)** | {gla_res['ma']:.1f} | {gla_res['prec']:.1f} | {gla_res['rec']:.1f} | {gla_res['f1']:.1f} |\n")
        f.write(f"| **Balo/Túi (Backpack)** | {bag_res['ma']:.1f} | {bag_res['prec']:.1f} | {bag_res['rec']:.1f} | {bag_res['f1']:.1f} |\n\n")
        f.write("### 3. Chỉ số Nhận dạng Màu sắc (Color)\n\n")
        f.write(f"- **Màu áo (Upper Color Acc):** **{upper_acc:.2f}%**\n")
        f.write(f"- **Màu quần (Lower Color Acc):** **{lower_acc:.2f}%**\n")
        f.write(f"- **Độ chính xác màu trung bình:** **{avg_color_acc:.2f}%**\n")
    print(f"[+] Đã xuất báo cáo Markdown: {md_path}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá thực nghiệm video PDR-System")
    parser.add_argument("--video", required=True, help="Đường dẫn file video test")
    parser.add_argument("--gt", required=True, help="Đường dẫn file Ground Truth CSV")
    parser.add_argument("--output_dir", default="results/benchmarks", help="Thư mục lưu kết quả")
    parser.add_argument("--tracker", default="bytetrack", help="Tên tracker (bytetrack, botsort)")
    parser.add_argument("--iou", type=float, default=0.5, help="Ngưỡng IoU so khớp (mặc định 0.5)")
    args = parser.parse_args()

    evaluate_video(
        video_path=args.video,
        gt_path=args.gt,
        output_dir=args.output_dir,
        iou_threshold=args.iou,
        tracker_name=args.tracker
    )
