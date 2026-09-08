import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

def create_presentation():
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(base_dir, 'docs', 'PDR_System_Defense_Slide.pptx')

    # Create Presentation
    prs = Presentation()

    # Apply a modern, clean template-like theme manually
    
    # 1. Title Slide
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]

    title.text = "Hệ thống Phát hiện & Tìm kiếm Người\n(PDR-System)"
    subtitle.text = "Dựa trên Đặc điểm Nhận dạng Trực quan\n\nNgười thực hiện: Nguyễn Đức Tài Năng"

    # Format Title
    title.text_frame.paragraphs[0].font.size = Pt(44)
    title.text_frame.paragraphs[0].font.bold = True
    title.text_frame.paragraphs[0].font.color.rgb = RGBColor(0, 51, 153) # Deep Blue
    
    # 2. Vấn Đề Slide
    bullet_slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(bullet_slide_layout)
    shapes = slide.shapes
    title_shape = shapes.title
    body_shape = shapes.placeholders[1]

    title_shape.text = "1. Đặt Vấn Đề"
    tf = body_shape.text_frame
    tf.text = "Khó khăn trong hệ thống giám sát truyền thống:"
    
    p = tf.add_paragraph()
    p.text = "Camera an ninh chỉ ghi hình, khó tìm kiếm theo yêu cầu cụ thể (màu áo, giới tính, phụ kiện)."
    p.level = 1

    p = tf.add_paragraph()
    p.text = "Phụ thuộc hoàn toàn vào con người dò tìm thủ công."
    p.level = 1

    p = tf.add_paragraph()
    p.text = "Giải pháp: Ứng dụng AI/Deep Learning để tự động hóa quá trình trích xuất và tìm kiếm đối tượng."
    p.level = 0
    p.font.bold = True
    p.font.color.rgb = RGBColor(0, 128, 0)

    # 3. Kiến Trúc Slide
    slide = prs.slides.add_slide(bullet_slide_layout)
    shapes = slide.shapes
    title_shape = shapes.title
    body_shape = shapes.placeholders[1]

    title_shape.text = "2. Kiến Trúc Hệ Thống"
    tf = body_shape.text_frame
    tf.text = "Hệ thống bao gồm 3 Module chính:"
    
    p = tf.add_paragraph()
    p.text = "1. Object Detection & Tracking (YOLOv8n + ByteTrack): Phát hiện người và theo dõi quỹ đạo (Track ID)."
    p.level = 1

    p = tf.add_paragraph()
    p.text = "2. Feature Extraction (ResNet50 PAR): Trích xuất đặc điểm (Giới tính, Nón, Mắt kính, Balo) từ ảnh crop."
    p.level = 1

    p = tf.add_paragraph()
    p.text = "3. Color Detection (HSV Trimmed Median): Nhận diện màu áo bằng phương pháp cắt biên trung vị siêu tốc."
    p.level = 1

    # 4. Tối ưu hiệu năng
    slide = prs.slides.add_slide(bullet_slide_layout)
    shapes = slide.shapes
    title_shape = shapes.title
    body_shape = shapes.placeholders[1]

    title_shape.text = "3. Tối Ưu Hiệu Năng (Load Budgeting)"
    tf = body_shape.text_frame
    tf.text = "Giải quyết nút thắt cổ chai của ResNet50 (chạy rất nặng):"
    
    p = tf.add_paragraph()
    p.text = "Giới hạn số lượng CNN chạy mỗi frame (Load Budgeting) -> Đảm bảo FPS luôn mượt."
    p.level = 1

    p = tf.add_paragraph()
    p.text = "Sử dụng EMA (Exponential Moving Average) để làm mượt kết quả dự đoán qua từng frame."
    p.level = 1

    p = tf.add_paragraph()
    p.text = "Thay K-Means bằng thuật toán màu HSV Trimmed Median -> Giảm thời gian xử lý màu từ 1.7s xuống 0.1ms."
    p.level = 1

    # 5. Kết Quả Slide
    slide = prs.slides.add_slide(bullet_slide_layout)
    shapes = slide.shapes
    title_shape = shapes.title
    body_shape = shapes.placeholders[1]

    title_shape.text = "4. Kết Quả Đánh Giá"
    tf = body_shape.text_frame
    tf.text = "Đánh giá mô hình ResNet50 trên tập PA-100K:"
    
    p = tf.add_paragraph()
    p.text = "Mean Accuracy (mA): ~89.33%"
    p.level = 1

    p = tf.add_paragraph()
    p.text = "Balo (Backpack): 96.70%"
    p.level = 2

    p = tf.add_paragraph()
    p.text = "Mắt kính (Glasses): 91.00%"
    p.level = 2

    p = tf.add_paragraph()
    p.text = "Hiệu năng hệ thống đạt ~70 FPS trên GPU (GTX 1650), ~20 FPS trên CPU (i5)."
    p.level = 1

    # Save
    prs.save(output_path)
    print(f"Success: PPTX Presentation created at {output_path}")

if __name__ == "__main__":
    create_presentation()
