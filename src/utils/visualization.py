"""
src/utils/visualization.py
============================
Module vẽ bounding box, nhãn tiếng Việt và thông tin lên frame video.

LUỒNG XỬ LÝ:
    Frame gốc (BGR)
         │
         ├── draw_detections()    → Vẽ tất cả bounding box
         ├── draw_person_info()   → Vẽ thuộc tính + điểm số tiếng Việt từng người
         └── draw_query_panel()   → Vẽ panel mục tiêu tìm kiếm tiếng Việt
         │
    Frame đã annotate (BGR)
"""

import cv2
import numpy as np


# ── Bảng màu cho bounding box ──────────────────────────────
COLOR_MATCHED   = (0, 255, 0)     # BGR: Xanh lá (Đúng mục tiêu tìm kiếm)
COLOR_NOT_MATCH = (180, 180, 180) # BGR: Xám (Người khác)
COLOR_ANALYZING = (0, 165, 255)   # BGR: Cam
COLOR_TEXT_BG   = (0, 0, 0)       # BGR: Đen (nền text)
COLOR_WHITE     = (255, 255, 255)
COLOR_YELLOW    = (0, 255, 255)

# Bảng dịch màu sang tiếng Việt chuẩn có dấu
COLOR_VI_MAP = {
    "black": "Đen", "white": "Trắng", "red": "Đỏ", "blue": "Xanh dương",
    "green": "Xanh lá", "yellow": "Vàng", "gray": "Xám", "purple": "Tím", "other": "Khác"
}


def translate_color(color_name: str) -> str:
    if not color_name: return "Khác"
    return COLOR_VI_MAP.get(str(color_name).lower(), str(color_name))


def draw_bounding_box(frame, bbox, track_id, color=COLOR_NOT_MATCH, thickness=2):
    """
    Vẽ bounding box cho một người.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]

    # Vẽ hình chữ nhật
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Vẽ nhãn Track ID ở góc trên
    label = f"ID #{track_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    label_size, baseline = cv2.getTextSize(label, font, font_scale, 1)
    label_w, label_h = label_size

    # Nền cho text
    cv2.rectangle(
        frame,
        (x1, max(0, y1 - label_h - 8)),
        (x1 + label_w + 6, max(0, y1)),
        color, -1
    )
    # Text màu đen trên nền màu
    cv2.putText(
        frame, label,
        (x1 + 3, max(label_h + 2, y1 - 4)),
        font, font_scale, COLOR_TEXT_BG, 1, cv2.LINE_AA
    )

    return frame


def draw_person_info(
    frame,
    bbox,
    info_dict: dict = None,
    matched: bool = False,
    track_id: int = None,
    attributes: dict = None,
    score: float = None,
    is_matched: bool = None,
    **kwargs
):
    """
    Vẽ thông tin thuộc tính bằng tiếng Việt của người bên cạnh bounding box.
    Hỗ trợ cả định dạng info_dict hoặc tham số tường minh (track_id, attributes, score, is_matched).
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]

    if is_matched is not None:
        matched = is_matched

    # Gom thông tin nếu truyền dạng tường minh
    data = {}
    if info_dict is not None and isinstance(info_dict, dict):
        data.update(info_dict)
    if attributes is not None and isinstance(attributes, dict):
        data.update(attributes)
    if track_id is not None:
        data["track_id"] = track_id
    if score is not None:
        data["score"] = score

    color = COLOR_MATCHED if matched else COLOR_NOT_MATCH

    # Vẽ bbox
    draw_bounding_box(frame, bbox, data.get("track_id", "?"), color)

    if not matched:
        return frame

    # ── Xây dựng danh sách text tiếng Việt hiển thị ──
    lines = []

    gender_raw = str(data.get("gender", "?"))
    gender_vi = "Nu" if gender_raw.lower() in ["female", "nu", "nữ"] else "Nam"
    lines.append(f"Gioi tinh: {gender_vi}")

    upper = translate_color(data.get("upper_color", ""))
    lines.append(f"Ao : {upper}")

    lower = translate_color(data.get("lower_color", ""))
    lines.append(f"Quan: {lower}")

    accessories = []
    if data.get("hat"):      accessories.append("Mu")
    if data.get("glasses"):  accessories.append("Kinh")
    if data.get("backpack"): accessories.append("Balo")
    if accessories:
        lines.append(f"Kem: {', '.join(accessories)}")

    score = data.get("score")
    if score is not None:
        lines.append(f"Do khop: {score*100:.0f}%")

    # ── Vẽ text bên phải bbox ──
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    line_height = 18
    padding = 6

    max_w = max(cv2.getTextSize(l, font, font_scale, 1)[0][0] for l in lines)
    total_h = len(lines) * line_height + padding * 2

    frame_w = frame.shape[1]
    text_x = x2 + 5
    if text_x + max_w + padding * 2 > frame_w:
        text_x = max(0, x1 - max_w - padding * 2 - 5)

    text_y_start = max(0, y1)

    # Nền đen mờ
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (text_x, text_y_start),
        (text_x + max_w + padding * 2, text_y_start + total_h),
        (20, 20, 20), -1
    )
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Viền xanh lá
    cv2.rectangle(
        frame,
        (text_x, text_y_start),
        (text_x + max_w + padding * 2, text_y_start + total_h),
        COLOR_MATCHED, 1
    )

    # In từng dòng text
    for i, line in enumerate(lines):
        y_pos = text_y_start + padding + (i + 1) * line_height
        text_color = COLOR_YELLOW if line.startswith("Do khop") else COLOR_WHITE
        cv2.putText(
            frame, line,
            (text_x + padding, y_pos),
            font, font_scale, text_color, 1, cv2.LINE_AA
        )

    return frame


def draw_query_panel(frame, query: dict, threshold: float):
    """
    Vẽ panel góc trên phải hiển thị mục tiêu tìm kiếm tiếng Việt.
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    lines = ["-- MUC TIEU TIM KIEM --"]
    if query.get("gender") and query["gender"] not in ["Any", "Tùy ý (Bất kỳ)", "Tuy y"]:
        g_vi = "Nu" if str(query["gender"]).lower() in ["female", "nữ", "nu"] else "Nam"
        lines.append(f"Gioi tinh : {g_vi}")
    if query.get("upper_color") and query["upper_color"] not in ["Any", "Tùy ý (Bất kỳ)", "Tuy y"]:
        lines.append(f"Mau ao    : {translate_color(query['upper_color'])}")
    if query.get("lower_color") and query["lower_color"] not in ["Any", "Tùy ý (Bất kỳ)", "Tuy y"]:
        lines.append(f"Mau quan  : {translate_color(query['lower_color'])}")
    if query.get("backpack") is True:
        lines.append("Balo      : Co")
    if query.get("hat") is True:
        lines.append("Doi mu    : Co")
    if query.get("glasses") is True:
        lines.append("Deo kinh  : Co")
    lines.append(f"Nguong loc: {threshold*100:.0f}%")

    font_scale = 0.45
    line_height = 18
    padding = 8
    panel_w = 210
    panel_h = len(lines) * line_height + padding * 2

    # Vị trí: góc trên phải
    px = w - panel_w - 10
    py = 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (20, 20, 50), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), (100, 150, 255), 1)

    for i, line in enumerate(lines):
        y_pos = py + padding + (i + 1) * line_height
        color = COLOR_YELLOW if i == 0 else COLOR_WHITE
        cv2.putText(frame, line, (px + padding, y_pos), font, font_scale, color, 1, cv2.LINE_AA)

    return frame


def draw_fps_and_count(frame, fps: float, total_detected: int, total_matched: int):
    """
    Vẽ FPS và số lượng đối tượng theo dõi/khớp bằng tiếng Việt.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        f"FPS: {fps:.1f}",
        f"Dang theo doi: {total_detected}",
        f"Khop muc tieu: {total_matched}",
    ]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (180, 75), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    for i, line in enumerate(lines):
        color = COLOR_MATCHED if "Khop" in line and total_matched > 0 else COLOR_WHITE
        cv2.putText(frame, line, (8, 20 + i * 20), font, 0.5, color, 1, cv2.LINE_AA)

    return frame
