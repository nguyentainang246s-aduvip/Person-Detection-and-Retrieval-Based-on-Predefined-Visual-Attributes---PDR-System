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

def generate_docx_report(output_dir):
    doc = Document()
    
    # Title
    title = doc.add_heading('BÁO CÁO AUDIT & ĐẠI PHẪU HỆ THỐNG PDR', 0)
    title.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    
    doc.add_paragraph(f'Ngày xuất báo cáo: {datetime.datetime.now().strftime("%d/%m/%Y")}')
    doc.add_heading('1. Tổng quan dự án', level=1)
    doc.add_paragraph('Dự án PDR-System (Person Detection & Retrieval) sử dụng kiến trúc YOLOv8, ByteTrack và ResNet50 để phát hiện và tìm kiếm đối tượng người theo đặc điểm màu sắc và phụ kiện.')
    
    doc.add_heading('2. Các cải tiến cốt lõi (Refactoring)', level=1)
    
    p1 = doc.add_paragraph()
    p1.add_run('Color Detector: ').bold = True
    p1.add_run('Thay thế Trimmed Median bằng thuật toán phân cụm K-Means kết hợp cân bằng trắng (Gray-world normalization) giúp chống nhiễu màu trong điều kiện ánh sáng phức tạp.')
    
    p2 = doc.add_paragraph()
    p2.add_run('Soft Matching: ').bold = True
    p2.add_run('Cho phép khoan dung màu sắc (vd: Xanh đậm vs Đen) để không bị mất mục tiêu khi AI nhận dạng chệch một tông màu.')
    
    p3 = doc.add_paragraph()
    p3.add_run('Load Budgeting (Chống tràn RAM): ').bold = True
    p3.add_run('Tự động dọn rác các Track ID cũ (Garbage Collection) và ưu tiên phân tích AI cho các người mới xuất hiện (chống nạn đói CNN).')
    
    p4 = doc.add_paragraph()
    p4.add_run('Batch Inference: ').bold = True
    p4.add_run('Hỗ trợ PyTorch batching cho ResNet50, giảm overhead của Python khi chạy nhiều crop cùng một frame.')
    
    # Inject actual metrics if available
    eval_file = "results/evaluation/par_eval_results.json"
    if os.path.exists(eval_file):
        with open(eval_file, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        doc.add_heading('3. Đánh giá độ chính xác (Metrics)', level=1)
        
        if not eval_data.get('is_real_measurement', True):
            p_warn = doc.add_paragraph()
            p_warn.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
            run_warn = p_warn.add_run("⚠️ SỐ LIỆU DEMO — CHƯA ĐO THẬT ⚠️")
            run_warn.bold = True
            run_warn.font.color.rgb = RGBColor(255, 0, 0)
            run_warn.font.size = Pt(16)

        doc.add_paragraph(f"Đánh giá trên thiết bị: {eval_data.get('device', 'Unknown')}")
        mA = eval_data.get('mA', 0)
        doc.add_paragraph(f"Mean Accuracy (mA) trên PA-100K test set đạt {mA*100:.2f}%.")
        
        table = doc.add_table(rows=1, cols=2)
        table.style = 'Table Grid'
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = 'Thuộc tính'
        hdr_cells[1].text = 'Accuracy (%)'
        
        per_class = eval_data.get('per_class', {})
        for attr, vals in per_class.items():
            row_cells = table.add_row().cells
            row_cells[0].text = attr
            row_cells[1].text = f"{vals.get('accuracy', 0)*100:.2f}%"

    file_path = os.path.join(output_dir, 'PDR_Audit_Report.docx')
    doc.save(file_path)
    return file_path

def generate_pptx_slides(output_dir):
    prs = Presentation()
    
    # Title Slide
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    title.text = "BẢO VỆ ĐỒ ÁN: PDR-SYSTEM"
    subtitle.text = "Person Detection & Retrieval Based on Predefined Visual Attributes\n" + datetime.datetime.now().strftime("%B %Y")
    
    # Slide 1: Kiến trúc hệ thống
    bullet_slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(bullet_slide_layout)
    shapes = slide.shapes
    title_shape = shapes.title
    body_shape = shapes.placeholders[1]
    title_shape.text = "1. Kiến trúc Hệ Thống"
    tf = body_shape.text_frame
    tf.text = "Luồng xử lý thời gian thực:"
    p = tf.add_paragraph()
    p.text = "YOLOv8: Phát hiện bounding box người."
    p.level = 1
    p = tf.add_paragraph()
    p.text = "ByteTrack: Đảm bảo Track ID liên tục, chống mất dấu."
    p.level = 1
    p = tf.add_paragraph()
    p.text = "ResNet50 PAR & K-Means: Rút trích giới tính, mũ, kính, balo, màu sắc."
    p.level = 1

    # Slide 2: Tối ưu hiệu năng
    slide = prs.slides.add_slide(bullet_slide_layout)
    shapes = slide.shapes
    title_shape = shapes.title
    body_shape = shapes.placeholders[1]
    title_shape.text = "2. Tối Ưu Hiệu Năng (Refactor)"
    tf = body_shape.text_frame
    tf.text = "Các điểm nhấn kỹ thuật:"
    p = tf.add_paragraph()
    p.text = "Quản lý luồng (Load Budgeting) chặn tối đa 2 CNN / frame."
    p.level = 1
    p = tf.add_paragraph()
    p.text = "Soft Matching Engine: Tăng điểm recall khi điều kiện sáng kém."
    p.level = 1
    p = tf.add_paragraph()
    p.text = "GPU Batch Inference cho mạng CNN để khai thác VRAM hiệu quả."
    p.level = 1

    file_path = os.path.join(output_dir, 'PDR_Defense_Deck.pptx')
    prs.save(file_path)
    return file_path

def generate_xlsx_dashboard(output_dir):
    # P0-1: Đọc dữ liệu benchmark từ file JSON thực tế
    benchmark_file = "results/benchmark_results.json"
    if not os.path.exists(benchmark_file):
        raise FileNotFoundError(f"Chưa có số liệu đo đạc thực tế. Vui lòng chạy 'python scripts/benchmark_fps.py' để tạo {benchmark_file} trước khi xuất báo cáo.")
        
    with open(benchmark_file, "r", encoding="utf-8") as f:
        bench_data = json.load(f)
        
    file_path = os.path.join(output_dir, 'PDR_Evaluation_Dashboard.xlsx')
    workbook = xlsxwriter.Workbook(file_path)
    worksheet = workbook.add_worksheet('System Metrics')
    
    # Formats
    header_fmt = workbook.add_format({'bold': True, 'bg_color': '#4F81BD', 'font_color': 'white'})
    
    # Write Headers
    headers = ['Thành phần', 'Độ trễ trung bình (ms)', 'Tối ưu hóa', 'Tốc độ khung hình kỳ vọng']
    for col, h in enumerate(headers):
        worksheet.write(0, col, h, header_fmt)
        worksheet.set_column(col, col, 25)
    
    # Đọc data từ JSON map vào bảng
    # Cấu trúc: [Name, latency_ms, optimization_note, expected_fps_string]
    # Dùng list các dict từ json_data["modules"]
    modules = bench_data.get("modules", [])
    
    def get_latency(mod_name, default=0.0):
        for m in modules:
            if mod_name.lower() in m["name"].lower():
                return m.get("latency_ms", default)
        return default
        
    data = [
        ['YOLOv8 Detection', get_latency("YOLOv8"), 'TensorRT / Half Precision', '30 FPS'],
        ['ByteTrack MOT', get_latency("ByteTrack"), 'Matrix Operations', '30 FPS'],
        ['K-Means Color HSV', get_latency("Color"), 'Crop ROI xám hóa (Gray-world)', '> 60 FPS'],
        ['ResNet50 PAR', get_latency("ResNet50"), 'Batch Inference (2 crops/batch)', '25 FPS'],
        ['Overall Pipeline', get_latency("Pipeline"), 'Load Budgeting (Limit 2 CNN)', 'Real-time (25+ FPS)']
    ]
    
    for row_idx, row_data in enumerate(data, start=1):
        for col_idx, cell_data in enumerate(row_data):
            worksheet.write(row_idx, col_idx, cell_data)
            
    # Inject actual metrics if available to check if demo
    eval_file = "results/evaluation/par_eval_results.json"
    if os.path.exists(eval_file):
        with open(eval_file, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        if not eval_data.get('is_real_measurement', True):
            warn_fmt = workbook.add_format({'bold': True, 'font_color': 'red', 'font_size': 14})
            worksheet.merge_range('A7:E7', "⚠️ SỐ LIỆU DEMO — CHƯA ĐO THẬT ⚠️", warn_fmt)

    # Chart
    chart = workbook.add_chart({'type': 'column'})
    chart.add_series({
        'name': 'Độ trễ (ms)',
        'categories': ['System Metrics', 1, 0, 5, 0],
        'values': ['System Metrics', 1, 1, 5, 1],
    })
    chart.set_title({'name': f'Độ trễ xử lý theo từng module ({bench_data.get("device", "Unknown")})'})
    chart.set_y_axis({'name': 'Milliseconds (ms)'})
    worksheet.insert_chart('A9', chart)
    
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
