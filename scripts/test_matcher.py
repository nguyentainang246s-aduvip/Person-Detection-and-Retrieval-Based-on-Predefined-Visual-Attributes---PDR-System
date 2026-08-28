"""
scripts/test_matcher.py
=======================
PHASE 6 – CHECKPOINT & TEST SCRIPT: MATCHING ENGINE & SEARCH SCORING

Kiểm tra thuật toán tính điểm tương đồng (Matching Score) và các kịch bản tìm kiếm:
1. Tìm theo Giới tính + Màu áo (Nam, Áo đen)
2. Tìm kèm phụ kiện (Nam, Áo đen, Balo: Có)
3. Tìm kiếm với thuộc tính không xác định ("Any")
4. Kiểm tra độ nhạy với các ngưỡng Threshold (70%, 80%, 90%)

CÁCH CHẠY:
    .\\venv\\Scripts\\python.exe scripts/test_matcher.py
"""

import sys
import os

# Thêm thư mục gốc vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.matcher import AttributeMatcher


def test_matching_scenarios():
    print("\n-- 1. Kiem tra cac kich ban Matching thuc te --")
    matcher = AttributeMatcher(default_threshold=0.7)

    # Dữ liệu 1 người thực tế được AI trích xuất:
    detected_person = {
        "gender": "Male",
        "upper_color": "Black",
        "lower_color": "Blue",
        "backpack": True,
        "hat": False,
        "glasses": False
    }

    print("  [Doi tuong thuc te duoc AI phat hien]:")
    print(f"    - Gioi tinh: {detected_person['gender']}")
    print(f"    - Ao: {detected_person['upper_color']}, Quan: {detected_person['lower_color']}")
    print(f"    - Balo: {'Co' if detected_person['backpack'] else 'Khong'}, Mu: {'Co' if detected_person['hat'] else 'Khong'}, Kinh: {'Co' if detected_person['glasses'] else 'Khong'}\n")

    scenarios = [
        # Scenario 1: Khớp 100% (Nam, Áo đen, Balo)
        {
            "name": "Kich ban 1: Query (Nam + Ao den + Balo)",
            "query": {"gender": "Male", "upper_color": "Black", "backpack": True},
            "min_expected_score": 0.99,
            "expected_match": True
        },
        # Scenario 2: Khớp 3/4 thuộc tính (Nam + Áo đen + Balo + Quần đen -> Quần thực tế là Blue)
        {
            "name": "Kich ban 2: Query (Nam + Ao den + Balo + Quan den)",
            "query": {"gender": "Male", "upper_color": "Black", "lower_color": "Black", "backpack": True},
            "min_expected_score": 0.70,
            "expected_match": True # Vì score ~74% >= 70% threshold
        },
        # Scenario 3: Không khớp (Nữ + Áo trắng)
        {
            "name": "Kich ban 3: Query (Nu + Ao trang)",
            "query": {"gender": "Female", "upper_color": "White"},
            "min_expected_score": 0.0,
            "expected_match": False
        },
        # Scenario 4: Query để "Any" cho mọi thuộc tính không quan tâm
        {
            "name": "Kich ban 4: Query (Chi tim nguoi Doi mu = Co, cac thuoc tinh khac = Any)",
            "query": {"gender": "Any", "upper_color": "Any", "hat": True},
            "min_expected_score": 0.0,
            "expected_match": False # Vì người thực tế không đội mũ (hat=False)
        }
    ]

    all_passed = True

    for sc in scenarios:
        matched, score, breakdown = matcher.is_match(sc["query"], detected_person)
        is_ok = (matched == sc["expected_match"])
        if not is_ok:
            all_passed = False

        status = "✓" if is_ok else "✗"
        print(f"  [{status}] {sc['name']}")
        print(f"      • Score dat duoc : {score * 100:.1f}%")
        print(f"      • Ket luan Target: {'TARGET FOUND ✅' if matched else 'NOT MATCH ❌'}")
        print(f"      • Chi tiet match : {breakdown}\n")

    return all_passed


def test_threshold_sensitivity():
    print("-- 2. Kiem tra do nhay cua nguong Threshold (60%, 75%, 90%) --")
    matcher = AttributeMatcher()

    person = {
        "gender": "Male",
        "upper_color": "Black",
        "lower_color": "Blue",
        "backpack": False
    }

    # Query: Nam + Ao den + Quan den (Khop 2 tren 3)
    query = {"gender": "Male", "upper_color": "Black", "lower_color": "Black"}

    score, _ = matcher.calculate_score(query, person)
    print(f"  Query: Nam + Ao den + Quan den | Score thuc te: {score*100:.1f}%\n")

    thresholds = [0.60, 0.70, 0.85]
    for th in thresholds:
        m, _, _ = matcher.is_match(query, person, threshold=th)
        print(f"    - Nguong {th*100:.0f}% -> {'ACCEPTED (Hop le)' if m else 'REJECTED (Loai bo)'}")

    return True


def main():
    print("=" * 60)
    print("  PHASE 6 CHECKPOINT: MATCHING ENGINE & SEARCH SCORING")
    print("=" * 60)

    ok1 = test_matching_scenarios()
    ok2 = test_threshold_sensitivity()

    print("=" * 60)
    if ok1 and ok2:
        print("  [PASSED] PHASE 6 CHECKPOINT: HOAN THANH!")
        print("  -> San sang chuyen sang PHASE 7: Video Pipeline Integration & Demo!")
    else:
        print("  [FAILED] PHASE 6 CHECKPOINT: Co loi xay ra.")
    print("=" * 60)


if __name__ == "__main__":
    main()
