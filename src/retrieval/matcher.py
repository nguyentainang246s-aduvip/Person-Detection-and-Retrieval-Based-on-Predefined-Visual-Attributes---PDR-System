"""
Đề xuất cải tiến cho src/retrieval/matcher.py
==============================================
Thay đổi chính so với bản gốc:
1. So khớp màu theo kiểu "soft" (điểm 1.0 nếu trùng khớp, 0.5 nếu là màu "hàng xóm"
   trong không gian màu, 0.0 nếu khác hẳn) để giảm việc điểm số sụp về 0 chỉ vì
   ColorDetector lệch biên (Navy bị đọc thành Black chẳng hạn).
2. Bọc str() an toàn hơn, tránh crash nếu query truyền vào không phải string.
3. Thêm tham số min_confidence để loại bỏ các thuộc tính PAR có độ tin cậy quá thấp
   khỏi phép tính điểm (tránh một prediction "yếu" kéo điểm xuống oan).
"""

from src.utils.logger import get_logger

logger = get_logger("matcher")

# Các nhóm màu được coi là "gần nhau" -> phạt nhẹ thay vì phạt hoàn toàn
COLOR_NEIGHBORS = {
    "black": {"gray", "blue"},   # navy/xanh đậm hay bị đọc lẫn Black/Blue
    "gray": {"black", "white"},
    "blue": {"black", "purple"},
    "white": {"gray"},
}


class AttributeMatcher:
    DEFAULT_WEIGHTS = {
        "gender": 1.0,
        "upper_color": 1.5,
        "lower_color": 1.2,
        "backpack": 1.0,
        "hat": 1.0,
        "glasses": 0.8,
    }

    def __init__(self, weights: dict = None, default_threshold: float = 0.7,
                 min_confidence: float = 0.55, soft_color_score: float = 0.5):
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.default_threshold = default_threshold
        # Ngưỡng tin cậy tối thiểu để 1 thuộc tính PAR được tính vào điểm số
        self.min_confidence = min_confidence
        # Điểm "một phần" khi 2 màu là hàng xóm trong không gian màu (thay vì 0)
        self.soft_color_score = soft_color_score

    @staticmethod
    def _norm(v) -> str:
        return str(v).strip().lower() if v is not None else ""

    def _color_score(self, query_color: str, actual_color: str) -> float:
        q, a = self._norm(query_color), self._norm(actual_color)
        if not q or not a:
            return 0.0
        if q == a:
            return 1.0
        if a in COLOR_NEIGHBORS.get(q, set()):
            return self.soft_color_score
        return 0.0

    def calculate_score(self, target_query: dict, person_attributes: dict) -> tuple:
        if not target_query:
            return 1.0, {}

        total_weight_required = 0.0
        matched_weight = 0.0
        breakdown = {}

        # --- Gender ---
        q_gender = target_query.get("gender")
        if q_gender and self._norm(q_gender) != "any":
            w = self.weights.get("gender", 1.0)
            conf = person_attributes.get("gender_confidence", 1.0)
            if conf >= self.min_confidence:
                total_weight_required += w
                partial = 1.0 if self._norm(q_gender) == self._norm(person_attributes.get("gender")) else 0.0
                breakdown["gender"] = bool(partial)
                matched_weight += w * partial

        # --- Upper / Lower color (soft matching) ---
        for key in ("upper_color", "lower_color"):
            q_val = target_query.get(key)
            if q_val and self._norm(q_val) != "any":
                w = self.weights.get(key, 1.0)
                total_weight_required += w
                partial = self._color_score(q_val, person_attributes.get(key))
                breakdown[key] = partial  # float 0 / soft_color_score / 1.0, giữ chi tiết thay vì chỉ bool
                matched_weight += w * partial

        # --- Accessories (backpack / hat / glasses) ---
        for acc in ("backpack", "hat", "glasses"):
            q_acc = target_query.get(acc)
            if q_acc is not None and self._norm(q_acc) != "any":
                w = self.weights.get(acc, 1.0)
                conf = person_attributes.get(f"{acc}_confidence", 1.0)
                if conf < self.min_confidence:
                    # Dự đoán quá thiếu chắc chắn -> bỏ qua thay vì tính sai lệch
                    continue
                total_weight_required += w
                pred_val = bool(person_attributes.get(acc, False))
                is_match = (bool(q_acc) == pred_val)
                breakdown[acc] = is_match
                if is_match:
                    matched_weight += w

        if total_weight_required == 0.0:
            return 1.0, breakdown

        score = matched_weight / total_weight_required
        return round(score, 4), breakdown

    def is_match(self, target_query: dict, person_attributes: dict, threshold: float = None) -> tuple:
        thresh = threshold if threshold is not None else self.default_threshold
        score, breakdown = self.calculate_score(target_query, person_attributes)
        matched = score >= thresh
        return matched, score, breakdown
