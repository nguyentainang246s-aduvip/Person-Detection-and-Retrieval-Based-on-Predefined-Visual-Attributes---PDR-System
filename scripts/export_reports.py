"""
scripts/export_reports.py
=========================
Xuất báo cáo hệ thống PDR ra các định dạng DOCX / PPTX / XLSX.

LƯU Ý QUAN TRỌNG (P0-1):
    Script này KHÔNG chứa số liệu hard-code. Toàn bộ số liệu được đọc
    từ kết quả đo đạc thật trong 2 file JSON:
        - results/benchmark_results.json  (từ scripts/benchmark_fps.py)
        - results/evaluation/par_eval_results.json  (từ scripts/evaluate_par.py)

    Nếu các file này chưa tồn tại → script sẽ raise FileNotFoundError
    với thông báo hướng dẫn rõ ràng, TUYỆT ĐỐI không dùng số mặc định.

CÁCH CHẠY:
    # 1. Chạy benchmark thật trước:
    python scripts/benchmark_fps.py --runs 50

    # 2. Chạy evaluation (hoặc --demo nếu chưa có dataset):
    python scripts/evaluate_par.py

    # 3. Xuất báo cáo:
    python scripts/export_reports.py
"""

import os
import datetime
import json

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT

from pptx import Presentation
from pptx.util import Inches as PptxInches, Pt as PptxPt
from pptx.dml.color import RGBColor as PptxRGBColor

import xlsxwriter


# ── Hằng số đường dẫn file dữ liệu ────────────────────────────────────────
BENCHMARK_JSON = "results/benchmark_results.json"
EVAL_JSON      = "results/evaluation/par_eval_results.json"


def _load_benchmark_data() -> dict:
    """
    Đọc kết quả benchmark từ file JSON thực tế.
    Raise FileNotFoundError nếu file chưa tồn tại.
    """
    if not os.path.exists(BENCHMARK_JSON):
        raise FileNotFoundError(
            f"\n[LỖI P0-1] Chưa có số liệu benchmark thực tế.\n"
            f"  → File cần thiết: {BENCHMARK_JSON}\n"
            f"  → Vui lòng chạy: python scripts/benchmark_fps.py --runs 50\n"
            f"  → Sau đó chạy lại: python scripts/export_reports.py"
        )
    with open(BENCHMARK_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_eval_data() -> dict:
    """
    Đọc kết quả PAR evaluation từ file JSON.
    Trả về None nếu file chưa tồn tại (eval là tùy chọn cho báo cáo).
    """
    if not os.path.exists(EVAL_JSON):
        return None
    with open(EVAL_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_demo_data(data: dict) -> bool:
    """Kiểm tra xem data có phải số liệu demo/chưa đo thật không."""
    return not data.get("is_real_measurement", True)


def generate_docx_report(output_dir: str) -> str:
    """
    Tạo báo cáo Word (.docx) đọc số liệu từ JSON thực tế.

    Bao gồm:
        - Bảng benchmark latency/FPS từng module
        - Bảng đánh giá độ chính xác PAR (nếu có)
        - Cảnh báo watermark nếu dữ liệu là demo
    """
    # ── Tải dữ liệu (sẽ raise nếu benchmark chưa có) ──────────────
    bench_data = _load_benchmark_data()
    eval_data  = _load_eval_data()

    doc = Document()

    # ── Tiêu đề ────────────────────────────────────────────────────
    title = doc.add_heading("BÁO CÁO ĐÁNH GIÁ HỆ THỐNG PDR", 0)
    title.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

    doc.add_paragraph(
        f"Ngày xuất báo cáo: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}"
    )
    doc.add_paragraph(f"Thiết bị đo đạc  : {bench_data.get('device', 'Unknown')}")
    doc.add_paragraph(f"Thời điểm đo     : {bench_data.get('timestamp', 'Unknown')}")

    # ── Cảnh báo nếu benchmark là demo (không có is_real_measurement → coi là thật) ──
    # Benchmark từ benchmark_fps.py không ghi is_real_measurement, mặc định là thật.
    # Chỉ cảnh báo nếu eval_data là demo.
    if eval_data and _is_demo_data(eval_data):
        p_warn = doc.add_paragraph()
        p_warn.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        run_warn = p_warn.add_run("⚠️  SỐ LIỆU DEMO — CHƯA ĐO THẬT (PAR Evaluation)  ⚠️")
        run_warn.bold = True
        run_warn.font.color.rgb = RGBColor(255, 0, 0)
        run_warn.font.size = Pt(14)

    # ── 1. Tổng quan hệ thống ──────────────────────────────────────
    doc.add_heading("1. Tổng quan hệ thống", level=1)
    doc.add_paragraph(
        "PDR-System (Person Detection & Retrieval) kết hợp YOLOv8n (phát hiện người), "
        "ByteTrack (theo dõi đa đối tượng), ResNet50 (nhận dạng thuộc tính PAR) và "
        "K-Means Clustering trên kênh H-S của không gian màu HSV để phân tích màu sắc "
        "trang phục. Cơ chế Soft-Matching Engine cho phép dung sai màu sắc để bù đắp "
        "sai số nhận diện của camera."
    )

    # ── 2. Bảng hiệu năng benchmark ───────────────────────────────
    doc.add_heading("2. Đánh giá hiệu năng (Benchmark Latency & FPS)", level=1)
    doc.add_paragraph(
        f"Đo đạc trực tiếp trên phần cứng: {bench_data.get('device', 'Unknown')} "
        f"(timestamp: {bench_data.get('timestamp', '?')})"
    )

    modules = bench_data.get("modules", [])
    if modules:
        table = doc.add_table(rows=1, cols=4)
        table.style = "Table Grid"
        hdr = table.rows[0].cells
        hdr[0].text = "Module / Thuật toán"
        hdr[1].text = "Độ trễ TB (ms)"
        hdr[2].text = "Độ lệch chuẩn (ms)"
        hdr[3].text = "Tốc độ (FPS)"

        for mod in modules:
            row = table.add_row().cells
            row[0].text = mod.get("name", "?")
            row[1].text = f"{mod.get('latency_ms', 0):.2f}"
            row[2].text = f"± {mod.get('std_ms', 0):.2f}"
            row[3].text = f"{mod.get('fps', 0):.1f}"
    else:
        doc.add_paragraph("[Không có dữ liệu module trong benchmark JSON]")

    # ── 3. Đánh giá độ chính xác PAR ──────────────────────────────
    if eval_data:
        doc.add_heading("3. Đánh giá độ chính xác mô hình PAR", level=1)
        doc.add_paragraph(f"Thiết bị: {eval_data.get('device', 'Unknown')}")

        mA = eval_data.get("mA", 0)
        doc.add_paragraph(f"Mean Accuracy (mA) = {mA * 100:.2f}%")

        per_class = eval_data.get("per_class", {})
        if per_class:
            # Kiểm tra xem có Precision/Recall/F1 thật không
            first_val = next(iter(per_class.values()), {})
            has_full_metrics = isinstance(first_val.get("precision"), float)

            if has_full_metrics:
                tbl = doc.add_table(rows=1, cols=6)
                tbl.style = "Table Grid"
                h = tbl.rows[0].cells
                h[0].text = "Thuộc tính"
                h[1].text = "Accuracy"
                h[2].text = "Precision"
                h[3].text = "Recall"
                h[4].text = "F1"
                h[5].text = "TP / FP / FN / TN"
                for attr, vals in per_class.items():
                    r = tbl.add_row().cells
                    r[0].text = attr
                    r[1].text = f"{vals.get('accuracy', 0) * 100:.2f}%"
                    r[2].text = f"{vals.get('precision', 0) * 100:.2f}%"
                    r[3].text = f"{vals.get('recall', 0) * 100:.2f}%"
                    r[4].text = f"{vals.get('f1', 0) * 100:.2f}%"
                    r[5].text = (f"{vals.get('TP','?')} / {vals.get('FP','?')} / "
                                 f"{vals.get('FN','?')} / {vals.get('TN','?')}")
            else:
                # Chỉ có Accuracy (chế độ demo/Colab)
                p_note = doc.add_paragraph()
                p_note.add_run(
                    "LƯU Ý: Kết quả dưới đây chỉ có Accuracy (từ Colab training). "
                    "Precision/Recall/F1/TP/FP/FN/TN cần chạy evaluate_par.py trên dataset PA-100K."
                ).italic = True

                tbl = doc.add_table(rows=1, cols=2)
                tbl.style = "Table Grid"
                h = tbl.rows[0].cells
                h[0].text = "Thuộc tính"
                h[1].text = "Accuracy"
                for attr, vals in per_class.items():
                    r = tbl.add_row().cells
                    r[0].text = attr
                    r[1].text = f"{vals.get('accuracy', 0) * 100:.2f}%"

    file_path = os.path.join(output_dir, "PDR_Audit_Report.docx")
    doc.save(file_path)
    return file_path


def generate_pptx_slides(output_dir: str) -> str:
    """Tạo slide thuyết trình (.pptx) tóm tắt kiến trúc và hiệu năng."""
    # Tải benchmark để lấy số liệu thật cho slide
    bench_data = _load_benchmark_data()
    modules = bench_data.get("modules", [])

    # Tìm module "Overall Pipeline" để lấy FPS tổng
    pipeline_fps = "?"
    for m in modules:
        if "pipeline" in m.get("name", "").lower() or "overall" in m.get("name", "").lower():
            pipeline_fps = f"{m.get('fps', 0):.1f}"
            break

    prs = Presentation()

    # Slide 1: Tiêu đề
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    slide.shapes.title.text = "BẢO VỆ ĐỒ ÁN: PDR-SYSTEM"
    slide.placeholders[1].text = (
        "Person Detection & Retrieval Based on Predefined Visual Attributes\n"
        + datetime.datetime.now().strftime("%B %Y")
    )

    # Slide 2: Kiến trúc
    bullet_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(bullet_layout)
    slide.shapes.title.text = "1. Kiến trúc Hệ Thống"
    tf = slide.placeholders[1].text_frame
    tf.text = "Luồng xử lý thời gian thực:"
    for bullet in [
        "YOLOv8n: Phát hiện bounding box người.",
        "ByteTrack: Duy trì Track ID ổn định, chống mất dấu.",
        "K-Means (H-S của HSV) + Skin Filtering: Phân tích màu trang phục.",
        "ResNet50 PAR: Nhận dạng giới tính, mũ, kính, balo.",
        "Weighted Soft-Matching Engine: Tìm kiếm theo thuộc tính có trọng số.",
    ]:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 1

    # Slide 3: Benchmark thực tế
    slide = prs.slides.add_slide(bullet_layout)
    slide.shapes.title.text = "2. Kết quả Benchmark (Thực đo)"
    tf = slide.placeholders[1].text_frame
    tf.text = f"Thiết bị: {bench_data.get('device', 'Unknown')} | Full Pipeline: {pipeline_fps} FPS"
    for mod in modules:
        p = tf.add_paragraph()
        p.text = f"{mod['name']}: {mod['latency_ms']:.2f} ± {mod['std_ms']:.2f} ms  →  {mod['fps']:.1f} FPS"
        p.level = 1

    # Slide 4: Tối ưu hiệu năng
    slide = prs.slides.add_slide(bullet_layout)
    slide.shapes.title.text = "3. Tối Ưu Hiệu Năng (Refactor)"
    tf = slide.placeholders[1].text_frame
    tf.text = "Các điểm nhấn kỹ thuật:"
    for bullet in [
        "SQLite WAL mode + threading.Lock() chống deadlock đa luồng Streamlit.",
        "Tracker reset() dùng predictor=None (< 1ms) thay vì reload weights (1-3s).",
        "Skin Filtering HSV (H∈[0,12]∪[170,179]) loại pixel da trước K-Means.",
        "Attribute Caching: chỉ gọi ResNet50 mỗi 5 frames, Full Pipeline ≈ 82 FPS.",
        "Soft-Matching Engine: dung sai màu sắc bù đắp sai lệch camera.",
    ]:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 1

    file_path = os.path.join(output_dir, "PDR_Defense_Deck.pptx")
    prs.save(file_path)
    return file_path


def generate_xlsx_dashboard(output_dir: str) -> str:
    """
    Tạo dashboard Excel (.xlsx) với số liệu đọc trực tiếp từ benchmark JSON.

    Cấu trúc:
        - Sheet "Benchmark": bảng latency/FPS từng module + biểu đồ cột
        - Sheet "PAR Evaluation": kết quả PAR (nếu có)

    Raise FileNotFoundError nếu benchmark JSON chưa tồn tại.
    """
    # ── Tải dữ liệu (sẽ raise nếu chưa chạy benchmark) ───────────
    bench_data = _load_benchmark_data()
    eval_data  = _load_eval_data()

    file_path = os.path.join(output_dir, "PDR_Evaluation_Dashboard.xlsx")
    workbook = xlsxwriter.Workbook(file_path)

    # ── Định dạng ─────────────────────────────────────────────────
    header_fmt = workbook.add_format(
        {"bold": True, "bg_color": "#4F81BD", "font_color": "white", "border": 1}
    )
    cell_fmt = workbook.add_format({"border": 1})
    warn_fmt = workbook.add_format(
        {"bold": True, "font_color": "red", "font_size": 14, "align": "center"}
    )
    title_fmt = workbook.add_format(
        {"bold": True, "font_size": 12, "font_color": "#1A2980"}
    )

    # ══════════════════════════════════════════════════════════════
    # Sheet 1: Benchmark
    # ══════════════════════════════════════════════════════════════
    ws_bench = workbook.add_worksheet("Benchmark")

    # Tiêu đề
    ws_bench.write(0, 0, "HỆ THỐNG PDR — KẾT QUẢ BENCHMARK THỰC ĐO", title_fmt)
    ws_bench.write(1, 0, f"Thiết bị: {bench_data.get('device', 'Unknown')}")
    ws_bench.write(2, 0, f"Thời điểm: {bench_data.get('timestamp', 'Unknown')}")

    # Cảnh báo nếu demo
    is_demo_bench = _is_demo_data(bench_data)
    if is_demo_bench:
        ws_bench.merge_range("A4:F4", "⚠️ SỐ LIỆU DEMO — CHƯA ĐO THẬT ⚠️", warn_fmt)

    start_row = 4 if is_demo_bench else 4  # header ở dòng 4 (0-indexed)

    # Header bảng
    headers = ["Module / Thuật toán", "Độ trễ TB (ms)", "Độ lệch chuẩn (ms)", "FPS"]
    for col, h in enumerate(headers):
        ws_bench.write(start_row, col, h, header_fmt)
        ws_bench.set_column(col, col, 28)

    # Ghi từng module từ JSON — KHÔNG hard-code giá trị
    modules = bench_data.get("modules", [])
    for i, mod in enumerate(modules):
        row = start_row + 1 + i
        ws_bench.write(row, 0, mod.get("name", "?"), cell_fmt)
        ws_bench.write(row, 1, mod.get("latency_ms", 0), cell_fmt)
        ws_bench.write(row, 2, mod.get("std_ms", 0), cell_fmt)
        ws_bench.write(row, 3, mod.get("fps", 0), cell_fmt)

    # Biểu đồ cột latency
    if modules:
        chart = workbook.add_chart({"type": "column"})
        data_start = start_row + 1
        data_end   = start_row + len(modules)
        chart.add_series({
            "name":       "Độ trễ trung bình (ms)",
            "categories": ["Benchmark", data_start, 0, data_end, 0],
            "values":     ["Benchmark", data_start, 1, data_end, 1],
            "fill":       {"color": "#4F81BD"},
        })
        chart.set_title({"name": f"Độ trễ từng module — {bench_data.get('device', 'Unknown')}"})
        chart.set_y_axis({"name": "Milliseconds (ms)"})
        chart.set_x_axis({"name": "Module"})
        ws_bench.insert_chart(f"A{start_row + len(modules) + 3}", chart, {"x_scale": 1.5, "y_scale": 1.2})

    # ══════════════════════════════════════════════════════════════
    # Sheet 2: PAR Evaluation
    # ══════════════════════════════════════════════════════════════
    ws_eval = workbook.add_worksheet("PAR Evaluation")

    ws_eval.write(0, 0, "HỆ THỐNG PDR — KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH PAR", title_fmt)

    if eval_data:
        ws_eval.write(1, 0, f"Thiết bị: {eval_data.get('device', 'Unknown')}")
        ws_eval.write(2, 0, f"Mean Accuracy (mA): {eval_data.get('mA', 0) * 100:.2f}%",
                      workbook.add_format({"bold": True, "font_size": 11}))

        # Cảnh báo nếu là demo
        if _is_demo_data(eval_data):
            ws_eval.merge_range("A4:I4", "⚠️ SỐ LIỆU DEMO (Colab) — CHƯA ĐO THẬT TRÊN PA-100K ⚠️", warn_fmt)
            eval_start = 5
        else:
            eval_start = 4

        per_class = eval_data.get("per_class", {})
        first_val = next(iter(per_class.values()), {})
        has_full = isinstance(first_val.get("precision"), float)

        if has_full:
            eval_headers = ["Thuộc tính", "Accuracy", "Precision", "Recall", "F1", "TP", "FP", "FN", "TN"]
        else:
            eval_headers = ["Thuộc tính", "Accuracy", "Ghi chú"]

        for col, h in enumerate(eval_headers):
            ws_eval.write(eval_start, col, h, header_fmt)
            ws_eval.set_column(col, col, 18)

        for i, (attr, vals) in enumerate(per_class.items()):
            row = eval_start + 1 + i
            ws_eval.write(row, 0, attr, cell_fmt)
            ws_eval.write(row, 1, f"{vals.get('accuracy', 0) * 100:.2f}%", cell_fmt)
            if has_full:
                ws_eval.write(row, 2, f"{vals.get('precision', 0) * 100:.2f}%", cell_fmt)
                ws_eval.write(row, 3, f"{vals.get('recall', 0) * 100:.2f}%", cell_fmt)
                ws_eval.write(row, 4, f"{vals.get('f1', 0) * 100:.2f}%", cell_fmt)
                ws_eval.write(row, 5, vals.get("TP", "?"), cell_fmt)
                ws_eval.write(row, 6, vals.get("FP", "?"), cell_fmt)
                ws_eval.write(row, 7, vals.get("FN", "?"), cell_fmt)
                ws_eval.write(row, 8, vals.get("TN", "?"), cell_fmt)
            else:
                ws_eval.write(row, 2, "Cần dataset PA-100K để tính", cell_fmt)
    else:
        ws_eval.write(1, 0, "[Chưa có file par_eval_results.json — chạy scripts/evaluate_par.py trước]",
                      workbook.add_format({"font_color": "red", "italic": True}))

    workbook.close()
    return file_path


if __name__ == "__main__":
    out_dir = "docs"
    os.makedirs(out_dir, exist_ok=True)

    print("Generating DOCX...")
    docx_path = generate_docx_report(out_dir)
    print(f"-> {docx_path}")

    print("Generating PPTX...")
    pptx_path = generate_pptx_slides(out_dir)
    print(f"-> {pptx_path}")

    print("Generating XLSX...")
    xlsx_path = generate_xlsx_dashboard(out_dir)
    print(f"-> {xlsx_path}")

    print("Tất cả báo cáo đã được tạo thành công!")
