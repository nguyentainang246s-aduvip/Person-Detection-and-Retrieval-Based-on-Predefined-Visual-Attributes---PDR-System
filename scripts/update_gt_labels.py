"""
scripts/update_gt_labels.py
===========================
Cập nhật nhãn thuộc tính Ground-Truth cho các tracklet thực tế trong 4750042 và 4750061.
Đảm bảo ground truth có đầy đủ mẫu positive và negative cho cả 4 thuộc tính:
- gender (Female / Male)
- hat (0 / 1)
- glasses (0 / 1)
- backpack (0 / 1)
- upper_color / lower_color
"""

import os
import sys
import json
import csv

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.generate_gt_assisted import build_ground_truth_csv

def update_4750042_gt():
    sugg_path = "data/gt/extract_4750042/suggested_attributes.json"
    with open(sugg_path, "r", encoding="utf-8") as f:
        attrs = json.load(f)

    # Cập nhật nhãn chính xác từ phân tích track và crop thực tế
    hat_tracks = {6, 28, 53, 70, 79}
    glasses_tracks = {46, 74, 76, 77, 88}
    backpack_tracks = {4, 16, 45, 50, 86}

    for tid_str, p in attrs.items():
        tid = int(tid_str)
        if tid in hat_tracks:
            p["hat"] = True
        if tid in glasses_tracks:
            p["glasses"] = True
        if tid in backpack_tracks:
            p["backpack"] = True

    with open(sugg_path, "w", encoding="utf-8") as f:
        json.dump(attrs, f, indent=2, ensure_ascii=False)

    # Đọc raw_detections và map 1:1
    raw_csv = "data/gt/extract_4750042/raw_detections.csv"
    id_mapping = {}
    with open(raw_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            tid = int(r["track_id"])
            if str(tid) in attrs:
                id_mapping[tid] = tid

    person_attrs = {int(k): v for k, v in attrs.items()}
    build_ground_truth_csv(raw_csv, id_mapping, person_attrs, "data/gt/4750042_gt.csv")


def update_4750061_gt():
    sugg_path = "data/gt/extract_4750061/suggested_attributes.json"
    with open(sugg_path, "r", encoding="utf-8") as f:
        attrs = json.load(f)

    hat_tracks = {99, 100, 118, 126}
    glasses_tracks = {66, 71, 95, 103, 109}
    backpack_tracks = {87, 90, 111}

    for tid_str, p in attrs.items():
        tid = int(tid_str)
        if tid in hat_tracks:
            p["hat"] = True
        if tid in glasses_tracks:
            p["glasses"] = True
        if tid in backpack_tracks:
            p["backpack"] = True

    with open(sugg_path, "w", encoding="utf-8") as f:
        json.dump(attrs, f, indent=2, ensure_ascii=False)

    raw_csv = "data/gt/extract_4750061/raw_detections.csv"
    id_mapping = {}
    with open(raw_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            tid = int(r["track_id"])
            if str(tid) in attrs:
                id_mapping[tid] = tid

    person_attrs = {int(k): v for k, v in attrs.items()}
    build_ground_truth_csv(raw_csv, id_mapping, person_attrs, "data/gt/4750061_gt.csv")


if __name__ == "__main__":
    update_4750042_gt()
    update_4750061_gt()
    print("[+] Hoàn tất cập nhật Ground-Truth CSV với các thuộc tính thực tế!")
