"""
scripts/test_database.py
========================
PHASE 8 – CHECKPOINT & TEST SCRIPT: SQLITE DATABASE & SEARCH HISTORY

Kiểm tra các tính năng lưu trữ và truy vấn cơ sở dữ liệu:
1. Tạo bảng `queries` và `search_results`.
2. Ghi một phiên tìm kiếm mới (Save Query).
3. Ghi thông tin các đối tượng Target tìm thấy (Save Target Results).
4. Truy vấn lịch sử các phiên tìm kiếm gần đây (Get History).
5. Truy vấn chi tiết theo `query_id`.

CÁCH CHẠY:
    .\\venv\\Scripts\\python.exe scripts/test_database.py
"""

import sys
import os

# Thêm thư mục gốc vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.db import DatabaseManager


def test_database_operations():
    print("\n-- 1. Kiem tra khoi tao DatabaseManager --")
    test_db_path = "results/test_retrieval.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    db = DatabaseManager(db_path=test_db_path)
    print("  [✓] Khoi tao Database thanh cong!")

    print("\n-- 2. Kiem tra luu Query & Search Results vao Database --")
    query_1 = {
        "gender": "Male",
        "upper_color": "Black",
        "lower_color": "Black",
        "backpack": True,
        "hat": None,
        "glasses": None
    }
    q1_id = db.save_query(query_1, threshold=0.75)
    print(f"  [✓] Da luu Query #1 (ID: {q1_id})")

    # Lưu 2 target tìm thấy cho Query 1
    target_1 = {
        "track_id": 12,
        "timestamp": "00:01:25",
        "frame_idx": 750,
        "score": 0.92,
        "crop_path": "results/crops/person_12.jpg",
        "attributes": {
            "gender": "Male", "upper_color": "Black", "lower_color": "Black",
            "backpack": True, "hat": False, "glasses": False
        }
    }
    target_2 = {
        "track_id": 25,
        "timestamp": "00:02:10",
        "frame_idx": 1300,
        "score": 0.81,
        "crop_path": "results/crops/person_25.jpg",
        "attributes": {
            "gender": "Male", "upper_color": "Black", "lower_color": "Gray",
            "backpack": True, "hat": False, "glasses": False
        }
    }
    db.save_target_result(q1_id, target_1)
    db.save_target_result(q1_id, target_2)
    print(f"  [✓] Da luu 2 Target vao search_results cho Query #{q1_id}")

    # Tạo thêm 1 Query 2
    query_2 = {"gender": "Female", "upper_color": "White", "backpack": False}
    q2_id = db.save_query(query_2, threshold=0.80)
    print(f"  [✓] Da luu Query #2 (ID: {q2_id})")

    print("\n-- 3. Kiem tra truy van lich su tim kiem (Search History) --")
    history = db.get_recent_queries(limit=10)
    print(f"  [✓] Tong so phien tim kiem trong Database: {len(history)}")
    for q in history:
        print(f"      • Query #{q['id']} ({q['created_at']}): Gender={q['gender']}, Upper={q['upper_color']}, Threshold={q['threshold']*100:.0f}% -> Tim thay: {q['target_count']} targets")

    print(f"\n-- 4. Kiem tra truy van chi tiet cac Target cua Query #{q1_id} --")
    results = db.get_results_by_query_id(q1_id)
    print(f"  [✓] So luong Target lay ra: {len(results)}")
    for r in results:
        print(f"      • Track #{r['track_id']} | Score: {r['score']*100:.1f}% | Time: {r['timestamp_str']} | Ao: {r['upper_color']} | Balo: {'Co' if r['backpack'] else 'Khong'}")

    return len(history) == 2 and len(results) == 2


def main():
    print("=" * 60)
    print("  PHASE 8 CHECKPOINT: SQLITE DATABASE & SEARCH HISTORY")
    print("=" * 60)

    ok = test_database_operations()

    print("\n" + "=" * 60)
    if ok:
        print("  [PASSED] PHASE 8 CHECKPOINT: DATABASE HOAN THANH!")
        print("  -> San sang chuyen sang PHASE 9: STREAMLIT WEB GUI DEMO!")
    else:
        print("  [FAILED] PHASE 8 CHECKPOINT: Co loi thao tac database.")
    print("=" * 60)


if __name__ == "__main__":
    main()
