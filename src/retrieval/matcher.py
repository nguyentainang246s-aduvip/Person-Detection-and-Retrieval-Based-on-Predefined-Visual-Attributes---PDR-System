"""
src/retrieval/matcher.py
========================
Bộ máy tìm kiếm và tính điểm tương đồng (Attribute Matching Engine).

CÔNG THỨC TÍNH ĐIỂM HỌC THUẬT:
                     Tổng (Trọng số của các thuộc tính KHỚP)
    Score (%) = ─────────────────────────────────────────────────── × 100%
                Tổng (Trọng số của các thuộc tính ĐƯỢC YÊU CẦU)

QUY TẮC MATCHING:
    1. Thuộc tính có giá trị "Any" hoặc None:
       -> Người dùng KHÔNG yêu cầu -> KHÔNG tính vào mẫu số (không bị trừ điểm).
    2. Thuộc tính có yêu cầu cụ thể (ví dụ: gender="Male", upper_color="Black"):
       -> So khớp với thuộc tính thực tế của đối tượng.
       -> Khớp: +1 điểm (hoặc theo trọng số weight).
       -> Không khớp: 0 điểm.
"""

from src.utils.logger import get_logger

logger = get_logger("matcher")


class AttributeMatcher:
    """
    Bộ so khớp thuộc tính người dùng tìm kiếm với đặc điểm nhận diện của đối tượng.
    """

    # Trọng số mặc định cho từng thuộc tính (có thể tùy chỉnh)
    DEFAULT_WEIGHTS = {
        "gender": 1.0,
        "upper_color": 1.5,  # Màu áo thường nổi bật và dễ nhận biết nhất
        "lower_color": 1.2,  # Màu quần
        "backpack": 1.0,
        "hat": 1.0,
        "glasses": 0.8       # Kính phụ kiện nhỏ, trọng số điều chỉnh
    }

    def __init__(self, weights: dict = None, default_threshold: float = 0.7):
        """
        INPUT:
            weights (dict): Bảng trọng số cho từng thuộc tính
            default_threshold (float): Ngưỡng tương đồng tối thiểu để coi là Target Found (0.0 -> 1.0)
        """
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.default_threshold = default_threshold

    def calculate_score(self, target_query: dict, person_attributes: dict) -> tuple:
        """
        Tính toán điểm số matching chi tiết giữa Query và Người được phát hiện.

        INPUT:
            target_query (dict): Truy vấn tìm kiếm từ người dùng, ví dụ:
                {
                    "gender": "Male",         # "Male", "Female", "Any"
                    "upper_color": "Black",   # "Black", "White", ..., "Any"
                    "lower_color": "Black",   # "Black", "Blue", ..., "Any"
                    "backpack": True,         # True, False, None
                    "hat": None,              # None = Any
                    "glasses": None           # None = Any
                }
            person_attributes (dict): Thuộc tính trích xuất được từ AI:
                {
                    "gender": "Male",
                    "upper_color": "Black",
                    "lower_color": "Blue",
                    "backpack": True,
                    "hat": False,
                    "glasses": False
                }

        OUTPUT:
            score (float): Điểm tương đồng từ 0.0 đến 1.0 (ví dụ: 0.75 = 75%)
            breakdown (dict): Chi tiết từng thuộc tính khớp hay không khớp {attr: bool}
        """
        if not target_query:
            return 1.0, {} # Nếu không yêu cầu gì thì match 100%

        total_weight_required = 0.0
        matched_weight = 0.0
        breakdown = {}

        # 1. So khớp Gender
        q_gender = target_query.get("gender")
        if q_gender and q_gender != "Any":
            w = self.weights.get("gender", 1.0)
            total_weight_required += w
            is_match = (q_gender.lower() == str(person_attributes.get("gender", "")).lower())
            breakdown["gender"] = is_match
            if is_match:
                matched_weight += w

        # 2. So khớp Màu áo (Upper Color)
        q_upper = target_query.get("upper_color")
        if q_upper and q_upper != "Any":
            w = self.weights.get("upper_color", 1.0)
            total_weight_required += w
            is_match = (q_upper.lower() == str(person_attributes.get("upper_color", "")).lower())
            breakdown["upper_color"] = is_match
            if is_match:
                matched_weight += w

        # 3. So khớp Màu quần (Lower Color)
        q_lower = target_query.get("lower_color")
        if q_lower and q_lower != "Any":
            w = self.weights.get("lower_color", 1.0)
            total_weight_required += w
            is_match = (q_lower.lower() == str(person_attributes.get("lower_color", "")).lower())
            breakdown["lower_color"] = is_match
            if is_match:
                matched_weight += w

        # 4. So khớp Phụ kiện (Backpack, Hat, Glasses)
        for acc in ["backpack", "hat", "glasses"]:
            q_acc = target_query.get(acc)
            # Chỉ xét khi người dùng chọn rõ True (Có) hoặc False (Không), bỏ qua None/"Any"
            if q_acc is not None and q_acc != "Any":
                w = self.weights.get(acc, 1.0)
                total_weight_required += w
                pred_val = bool(person_attributes.get(acc, False))
                is_match = (bool(q_acc) == pred_val)
                breakdown[acc] = is_match
                if is_match:
                    matched_weight += w

        # Nếu người dùng không chọn bất kỳ tiêu chí nào
        if total_weight_required == 0.0:
            return 1.0, breakdown

        score = matched_weight / total_weight_required
        return round(score, 4), breakdown

    def is_match(self, target_query: dict, person_attributes: dict, threshold: float = None) -> tuple:
        """
        Kiểm tra xem người này có đạt chuẩn Target Found dựa trên ngưỡng threshold hay không.

        OUTPUT:
            matched (bool): True nếu score >= threshold
            score (float): Điểm số (0.0 -> 1.0)
            breakdown (dict): Bảng chi tiết từng thuộc tính
        """
        thresh = threshold if threshold is not None else self.default_threshold
        score, breakdown = self.calculate_score(target_query, person_attributes)
        matched = (score >= thresh)

        return matched, score, breakdown
