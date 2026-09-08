"""
scripts/evaluate_par.py
========================
Script đánh giá định lượng mô hình PAR (Person Attribute Recognition)
trên tập test của PA-100K dataset.

KẾT QUẢ ĐẦU RA:
    - Mean Accuracy (mA) tổng hợp
    - Per-class Accuracy, Precision, Recall, F1
    - Confusion-style breakdown (TP, FP, FN, TN) mỗi attribute
    - Lưu kết quả ra results/evaluation/par_eval_results.csv
    - In bảng đẹp ra terminal

CÁCH SỬ DỤNG:
    # Đánh giá trên PA-100K test set (cần dataset local):
    python scripts/evaluate_par.py --data-root datasets/PA100K

    # Chỉ chạy nhanh trên 500 ảnh test mẫu ngẫu nhiên:
    python scripts/evaluate_par.py --data-root datasets/PA100K --max-samples 500

    # Chỉ test model trên ảnh thư mục bất kỳ:
    python scripts/evaluate_par.py --image-dir data/test_images/

ĐIỀU KIỆN:
    - Dataset PA-100K phải có sẵn tại --data-root (có file .mat annotation)
    - Nếu không có dataset: chạy ở chế độ --demo để xem kết quả từ Colab
"""

import os
import sys
import csv
import json
import time
import argparse
import numpy as np
import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attributes.par_model import AttributeRecognizer
from src.utils.logger import get_logger

logger = get_logger("evaluate_par")


# ── Kết quả training từ Colab T4 (dùng khi không có dataset local) ──────────
COLAB_RESULTS = {
    "meta": {
        "dataset": "PA-100K",
        "model": "ResNet50 Fine-tuned",
        "train_samples": 90_000,
        "val_samples": 10_000,
        "epochs_trained": 20,
        "best_epoch": 18,
        "training_device": "Colab T4 GPU"
    },
    "overall": {
        "val_mA": 0.8933,
        "description": "Mean Accuracy trung bình trên 4 thuộc tính"
    },
    "per_class": {
        "gender_female": {
            "accuracy": 0.8520,
            "description": "Phân loại Giới tính (Female vs Male)",
            "note": "Có thể thấp hơn trong CCTV do trang phục che khuất"
        },
        "hat": {
            "accuracy": 0.8470,
            "description": "Nhận dạng đội Mũ",
            "note": "Threshold nâng lên 0.62 để tránh nhầm tóc đen"
        },
        "glasses": {
            "accuracy": 0.9100,
            "description": "Nhận dạng đeo Kính",
            "note": "Accuracy cao nhất nhờ đặc điểm rõ ràng"
        },
        "backpack": {
            "accuracy": 0.9670,
            "description": "Nhận dạng đeo Balo",
            "note": "Accuracy cao nhất do đặc trưng hình dạng mạnh"
        }
    }
}


def print_banner():
    print("=" * 70)
    print("  ĐÁNH GIÁ MÔ HÌNH PAR (Person Attribute Recognition)")
    print("  ResNet50 Fine-tuned trên PA-100K Dataset")
    print("=" * 70)


def evaluate_on_dataset(data_root: str, max_samples: int = None) -> dict:
    """
    Đánh giá model trên PA-100K test set nếu có dataset local.
    """
    try:
        import scipy.io
        import cv2
    except ImportError:
        logger.error("Cần scipy và opencv để đọc dataset. Cài: pip install scipy opencv-python")
        return None

    # Tìm file annotation PA-100K
    mat_file = None
    for candidate in [
        os.path.join(data_root, "annotation.mat"),
        os.path.join(data_root, "PA100k_annotation.mat"),
        os.path.join(data_root, "data_release.mat"),
    ]:
        if os.path.exists(candidate):
            mat_file = candidate
            break

    if not mat_file:
        logger.warning(f"Không tìm thấy file annotation .mat trong '{data_root}'")
        logger.warning("Vui lòng tải PA-100K annotation và giải nén vào thư mục datasets/PA100K/")
        return None

    logger.info(f"Đọc annotation từ: {mat_file}")
    data = scipy.io.loadmat(mat_file)

    # PA-100K annotation structure
    # Labels: [gender, hat, glasses, backpack, ...]
    # Test partition: train=80000, val=10000, test=10000
    try:
        test_images = [str(f[0]).strip() for f in data["test_images_name"].flatten()]
        test_labels = data["test_label"]  # (10000, 26+)
    except KeyError:
        try:
            test_images = [str(f[0]).strip() for f in data["test_file_list"].flatten()]
            test_labels = data["test_labels"]
        except KeyError:
            logger.error("Không thể đọc cấu trúc annotation file. Kiểm tra lại file .mat")
            return None

    # PA-100K label indices (theo paper): gender=0, hat=1, glasses=2, backpack=...
    LABEL_IDX = {"gender_female": 0, "hat": 1, "glasses": 2, "backpack": 4}

    if max_samples:
        indices = np.random.choice(len(test_images), min(max_samples, len(test_images)), replace=False)
        test_images = [test_images[i] for i in indices]
        test_labels = test_labels[indices]
        logger.info(f"Chạy evaluation trên {len(test_images)} samples ngẫu nhiên")
    else:
        logger.info(f"Chạy evaluation trên toàn bộ {len(test_images)} test samples")

    recognizer = AttributeRecognizer()
    image_dir = os.path.join(data_root, "data")

    results = {attr: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for attr in LABEL_IDX}
    n_processed = 0
    t_start = time.time()

    for idx, (img_name, gt_row) in enumerate(zip(test_images, test_labels)):
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        preds = recognizer.predict(img)

        for attr, col_idx in LABEL_IDX.items():
            gt = int(gt_row[col_idx]) if gt_row[col_idx] in [0, 1] else (1 if gt_row[col_idx] > 0 else 0)

            if attr == "gender_female":
                pred = 1 if preds["gender"] == "Female" else 0
            elif attr == "hat":
                pred = 1 if preds["hat"] else 0
            elif attr == "glasses":
                pred = 1 if preds["glasses"] else 0
            elif attr == "backpack":
                pred = 1 if preds["backpack"] else 0
            else:
                pred = 0

            if gt == 1 and pred == 1:
                results[attr]["TP"] += 1
            elif gt == 0 and pred == 1:
                results[attr]["FP"] += 1
            elif gt == 1 and pred == 0:
                results[attr]["FN"] += 1
            else:
                results[attr]["TN"] += 1

        n_processed += 1
        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (idx + 1) * (len(test_images) - idx - 1)
            logger.info(f"  [{idx+1}/{len(test_images)}] ETA: {eta:.0f}s")

    # Tính metrics
    metrics = {}
    accuracies = []

    for attr, counts in results.items():
        tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
        total = tp + fp + fn + tn
        acc = (tp + tn) / total if total > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        metrics[attr] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn
        }
        accuracies.append(acc)

    mA = round(float(np.mean(accuracies)), 4)

    return {
        "source": "live_evaluation",
        "n_samples": n_processed,
        "mA": mA,
        "per_class": metrics
    }


def print_results_table(results: dict, source: str = "live"):
    """In bảng kết quả đẹp ra terminal."""

    print("\n" + "=" * 70)
    if source == "live":
        print(f"  KẾT QUẢ ĐÁNH GIÁ THỰC TẾ ({results.get('n_samples', '?')} samples)")
    else:
        print("  KẾT QUẢ TỪ COLAB T4 GPU TRAINING (PA-100K)")
    print("=" * 70)

    attr_names_vi = {
        "gender_female": "Giới tính (Female)",
        "hat": "Đội Mũ",
        "glasses": "Đeo Kính",
        "backpack": "Đeo Balo"
    }

    if source == "colab":
        print(f"\n  Mean Accuracy (mA): {results['overall']['val_mA']*100:.2f}%")
        print(f"\n  {'Thuộc tính':<22} {'Accuracy':>10}")
        print("  " + "-" * 35)
        for attr, data in results["per_class"].items():
            name_vi = attr_names_vi.get(attr, attr)
            acc_str = f"{data['accuracy']*100:.2f}%"
            print(f"  {name_vi:<22} {acc_str:>10}")
            print(f"    → {data['note']}")
        print()
        print("  Dataset: PA-100K | Model: ResNet50 | Training: Colab T4 GPU")
        print("  Epochs: 20 | Best epoch: 18 | Train samples: 90,000")
    else:
        mA = results.get("mA", 0)
        print(f"\n  Mean Accuracy (mA): {mA*100:.2f}%")
        print(f"\n  {'Thuộc tính':<22} {'Accuracy':>10} {'Precision':>11} {'Recall':>8} {'F1':>8}")
        print("  " + "-" * 65)
        for attr, data in results["per_class"].items():
            name_vi = attr_names_vi.get(attr, attr)
            print(f"  {name_vi:<22} {data['accuracy']*100:>9.2f}% "
                  f"{data['precision']*100:>10.2f}% "
                  f"{data['recall']*100:>7.2f}% "
                  f"{data['f1']*100:>7.2f}%")

    print("\n" + "=" * 70)


def save_results_csv(results: dict, output_path: str):
    """
    Lưu kết quả evaluation ra CSV.

    Nếu kết quả từ Colab demo (không có Precision/Recall/F1/TP/FP/FN/TN),
    ghi chú rõ ràng thay vì để dấu '-' im lặng.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Attribute", "Accuracy", "Precision", "Recall", "F1", "TP", "FP", "FN", "TN"])

        for attr, data in results.get("per_class", {}).items():
            def fmt_pct(val):
                """Định dạng phần trăm nếu là số, giữ nguyên nếu là chuỗi ghi chú."""
                if isinstance(val, float):
                    return f"{val*100:.2f}%"
                if isinstance(val, int) and not isinstance(val, bool):
                    return f"{val*100:.2f}%"
                # Là chuỗi ghi chú — giữ nguyên để người đọc hiểu rõ
                return str(val)

            writer.writerow([
                attr,
                fmt_pct(data.get("accuracy", 0)),
                fmt_pct(data.get("precision", "Cần dataset PA-100K")),
                fmt_pct(data.get("recall",    "Cần dataset PA-100K")),
                fmt_pct(data.get("f1",        "Cần dataset PA-100K")),
                data.get("TP", "Cần dataset PA-100K"),
                data.get("FP", "Cần dataset PA-100K"),
                data.get("FN", "Cần dataset PA-100K"),
                data.get("TN", "Cần dataset PA-100K"),
            ])

        writer.writerow([])
        mA = results.get("mA") or results.get("overall", {}).get("val_mA", 0)
        writer.writerow(["Mean Accuracy (mA)", f"{mA*100:.2f}%"])
        # Ghi rõ nguồn gốc để tránh nhầm lẫn khi đọc CSV
        is_real = results.get("source", "colab") != "colab"
        writer.writerow(["Nguồn số liệu",
                         "Đo thật trên PA-100K test set" if is_real
                         else "Số liệu Accuracy từ Colab training — CHƯA có Precision/Recall/F1 (cần dataset PA-100K local)"])

    logger.info(f"Đã lưu kết quả evaluation ra: {output_path}")


def save_results_json(results: dict, output_path: str):
    """Lưu kết quả ra JSON format cho báo cáo."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    import torch
    device_name = "cuda (" + torch.cuda.get_device_name(0) + ")" if torch.cuda.is_available() else "cpu"
    if results.get("source") == "colab":
        device_name = "Colab T4 GPU (Demo)"

    json_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "device": device_name,
        "is_real_measurement": results.get("source") != "colab",
        "mA": results.get("mA") or results.get("overall", {}).get("val_mA", 0),
        "per_class": results.get("per_class", {})
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
        
    logger.info(f"Đã lưu kết quả JSON ra: {output_path}")


def main():
    print_banner()

    parser = argparse.ArgumentParser(description="Đánh giá PAR model trên PA-100K")
    parser.add_argument("--data-root", type=str, default="datasets/PA100K",
                        help="Đường dẫn thư mục dataset PA-100K")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Số samples test tối đa (None = toàn bộ)")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Thư mục ảnh bất kỳ để test (không cần annotation)")
    parser.add_argument("--demo", action="store_true",
                        help="Hiển thị kết quả từ Colab training (không cần dataset)")
    parser.add_argument("--output", type=str, default="results/evaluation/par_eval_results.csv",
                        help="Đường dẫn lưu file CSV kết quả")
    args = parser.parse_args()

    # ── Chế độ Demo (hiển thị kết quả Colab) ─────────────────────
    if args.demo or not os.path.exists(args.data_root):
        if not args.demo:
            logger.warning(f"Không tìm thấy dataset tại '{args.data_root}'")
            logger.info("Chuyển sang chế độ --demo (hiển thị kết quả Colab T4)")

        print_results_table(COLAB_RESULTS, source="colab")

        # Lưu Colab results ra CSV.
        # Precision/Recall/F1/TP/FP/FN/TN được để trống vì không có dataset local;
        # save_results_csv() sẽ ghi chú rõ "Cần dataset PA-100K" thay vì "-" im lặng.
        colab_csv = {
            "mA": COLAB_RESULTS["overall"]["val_mA"],
            "source": "colab",  # Đánh dấu nguồn để save_results_csv xử lý đúng
            "per_class": {
                attr: {
                    "accuracy": data["accuracy"]
                    # Không có precision/recall/f1/TP/FP/FN/TN → để missing key
                    # → save_results_csv sẽ dùng default "Cần dataset PA-100K"
                }
                for attr, data in COLAB_RESULTS["per_class"].items()
            }
        }
        save_results_csv(colab_csv, args.output)
        
        json_output = args.output.replace(".csv", ".json")
        if json_output == args.output:
            json_output = args.output + ".json"
        
        colab_csv["source"] = "colab"
        save_results_json(colab_csv, json_output)

        print(f"\n  [INFO] Kết quả đã lưu ra: {args.output} và {json_output}")
        print(f"  [INFO] Để chạy trên PA-100K test set thực: tải dataset về datasets/PA100K/")
        print(f"  [INFO] Sau đó chạy lại: python scripts/evaluate_par.py --data-root datasets/PA100K\n")
        return

    # ── Chế độ Evaluation thực tế ─────────────────────────────────
    logger.info("Bắt đầu Evaluation trên PA-100K test set...")
    results = evaluate_on_dataset(args.data_root, max_samples=args.max_samples)

    if results is None:
        logger.error("Evaluation thất bại. Thử chạy với --demo để xem kết quả Colab.")
        logger.info("  python scripts/evaluate_par.py --demo")
        sys.exit(1)

    print_results_table(results, source="live")
    save_results_csv(results, args.output)
    
    json_output = args.output.replace(".csv", ".json")
    if json_output == args.output:
        json_output = args.output + ".json"
    save_results_json(results, json_output)

    print(f"\n  [✓] Evaluation hoàn tất!")
    print(f"  [✓] Kết quả CSV : {args.output}")
    print(f"  [✓] Kết quả JSON: {json_output}\n")


if __name__ == "__main__":
    main()
