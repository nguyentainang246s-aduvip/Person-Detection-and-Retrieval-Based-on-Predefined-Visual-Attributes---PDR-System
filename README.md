# 🎯 Person Detection & Retrieval System (PDR-System)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2Bcu121-orange?logo=pytorch)
![YOLOv8](https://img.shields.io/badge/YOLO-v8n-green?logo=yolo)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_UI-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**Hệ thống phát hiện và tìm kiếm người trong video dựa trên thuộc tính ngoại hình định trước**

*Person Detection and Retrieval Based on Predefined Visual Attributes*

</div>

---

## 📖 Giới thiệu

**PDR-System** là hệ thống mã nguồn mở hỗ trợ **phát hiện, theo vết và truy vấn tìm kiếm người** trong video giám sát hoặc video tải lên dựa trên các thuộc tính ngoại hình định trước (giới tính, màu áo, màu quần, phụ kiện như mũ, kính, balo).

Hệ thống kết hợp các mô hình thị giác máy tính và học sâu tiên tiến (**YOLOv8 + ByteTrack + ResNet50 PAR + K-Means Color Clustering + Soft-Matching Engine**) để xử lý và hiển thị kết quả theo thời gian thực qua giao diện web trực quan.

---

## ✨ Tính năng nổi bật & Cải tiến cốt lõi

- 🚶 **Phát hiện người (Person Detection):** Tích hợp **YOLOv8n** tối ưu tốc độ và độ chính xác trên từng khung hình.
- 🎯 **Theo dõi đa đối tượng (MOT Tracking):** Thuật toán **ByteTrack** duy trì Track ID ổn định, giảm thiểu mất dấu khi đối tượng bị che khuất tạm thời (occlusion).
- 🎨 **Phân tích màu sắc K-Means thích ứng:** Sử dụng phân cụm **K-Means Clustering** trong không gian HSV/Lab, tự động loại bỏ nhiễu nền và xác định chính xác màu áo/quần.
- 🧠 **Nhận diện thuộc tính (PAR - ResNet50):** Nhận diện đa thuộc tính (giới tính, mũ, kính, balo) hỗ trợ **Batch Inference** và tăng tốc trên GPU CUDA.
- ⚡ **Kiểm soát tải (Load Budgeting & Frame Skipping):** Tối ưu hóa pipeline, chỉ trích xuất thuộc tính định kỳ hoặc khi tracklet mới xuất hiện, duy trì FPS mượt mà.
- ⚖️ **Bộ máy so khớp mềm (Soft-Matching & Weighted Scoring):** Tính điểm tương đồng dựa trên trọng số tin cậy và khoảng cách màu sắc Delta-E, hỗ trợ tìm kiếm linh hoạt (`Any`).
- 🖥️ **Giao diện Web tương tác (Streamlit):** Xem video trực tiếp, điều chỉnh bộ lọc tìm kiếm, hiển thị đối tượng khớp và lưu lịch sử vào SQLite.
- 📊 **Tự động xuất báo cáo học thuật & đồ án:** Hỗ trợ script xuất tự động báo cáo Word (.docx), slide bảo vệ PowerPoint (.pptx) và bảng đánh giá Excel (.xlsx).

---

## 🏗️ Luồng xử lý Pipeline

```
Video / Camera Input
         │
         ▼
YOLOv8n Person Detection     ──► Phát hiện bounding box người
         │
         ▼
ByteTrack MOT Tracking       ──► Gán Track ID ổn định qua các frame
         │
         ▼
ROI Crop & Load Budgeting    ──► Chọn lọc & gom lô (Batch) tracklet cần xử lý
         │
         ├───► K-Means Adaptive Color Clustering (~0.3ms)
         ├───► ResNet50 PAR Batch Inference (~4ms/batch trên GPU)
         └───► Temporal EMA Smoothing (α=0.35)
         │
         ▼
Soft-Matching Engine         ──► Chấm điểm tương đồng theo trọng số & màu Lab
         │
         ▼
Person Retrieval Result      ──► Highlight đối tượng & hiển thị trên Web UI
         │
         ▼
Streamlit Web UI + SQLite Database + Export Reports
```

---

## 📁 Cấu trúc thư mục dự án

```
├── app.py                    # Giao diện Web Streamlit
├── config/config.yaml        # Cấu hình tham số hệ thống
├── requirements.txt          # Danh sách thư viện phụ thuộc
│
├── src/
│   ├── detection/detector.py # Module phát hiện người (YOLOv8)
│   ├── tracking/tracker.py   # Module theo dõi đối tượng (ByteTrack)
│   ├── attributes/
│   │   ├── color_detector.py # Phân tích màu sắc (K-Means Clustering)
│   │   └── par_model.py      # Nhận diện thuộc tính PAR (ResNet50 + Batch)
│   ├── retrieval/
│   │   ├── matcher.py        # Bộ so khớp Soft-Matching & Weighted Scoring
│   │   └── pipeline.py       # Pipeline tích hợp toàn bộ luồng xử lý
│   ├── database/db.py        # Quản lý lưu trữ SQLite
│   └── utils/                # Tiện ích logging, video, vẽ visualization
│
├── models/
│   ├── yolo/yolov8n.pt       # Trọng số YOLOv8n
│   └── par/par_resnet50.pth  # Trọng số ResNet50 PAR
│
├── docs/                     # Tài liệu thiết kế & báo cáo nghiệm thu
│   ├── PDR_Audit_Report.docx         # Báo cáo kỹ thuật chi tiết (.docx)
│   ├── PDR_Defense_Deck.pptx         # Slide trình bày bảo vệ (.pptx)
│   └── PDR_Evaluation_Dashboard.xlsx # Bảng đánh giá số liệu (.xlsx)
│
├── scripts/
│   ├── export_reports.py     # Script tự động xuất bộ báo cáo DOCX, PPTX, XLSX
│   ├── evaluate_par.py       # Đánh giá độ chính xác mô hình PAR
│   ├── benchmark_fps.py      # Đo kiểm hiệu năng và FPS
│   └── demo_pipeline.py      # Chạy demo bằng dòng lệnh CLI
│
└── data/
    └── test_videos/          # Video mẫu kiểm thử
```

---

## 🚀 Hướng dẫn cài đặt và sử dụng

### 1. Yêu cầu môi trường
- Python 3.10 trở lên
- Card đồ họa NVIDIA (khuyến nghị để đạt FPS cao) hoặc CPU

### 2. Cài đặt

```bash
# 1. Clone repository
git clone https://github.com/nguyentainang246s-aduvip/Person-Detection-and-Retrieval-Based-on-Predefined-Visual-Attributes---PDR-System.git
cd Person-Detection-and-Retrieval-Based-on-Predefined-Visual-Attributes---PDR-System

# 2. Tạo và kích hoạt môi trường ảo
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

# 3. Cài đặt các gói phụ thuộc
pip install -r requirements.txt

# 4. Cài đặt PyTorch hỗ trợ GPU CUDA (nếu máy có GPU NVIDIA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Khởi chạy giao diện Web

```bash
streamlit run app.py
```
Mở trình duyệt tại địa chỉ: `http://localhost:8501`

### 4. Xuất bộ tài liệu & slide báo cáo

```bash
python scripts/export_reports.py
```
Các file báo cáo sẽ được tạo tự động tại thư mục `docs/`:
- `docs/PDR_Audit_Report.docx` (Báo cáo kỹ thuật)
- `docs/PDR_Defense_Deck.pptx` (Slide thuyết trình)
- `docs/PDR_Evaluation_Dashboard.xlsx` (Bảng số liệu kiểm thử)

### 5. Chạy thử nghiệm bằng CLI

```bash
# Tìm người Nam mặc áo Đen
python scripts/demo_pipeline.py --gender Male --upper-color Black

# Tìm người đội Mũ và đeo Balo
python scripts/demo_pipeline.py --hat --backpack
```

### 6. Đo kiểm hiệu năng (Benchmark)

```bash
python scripts/benchmark_fps.py
```

---

## ⚙️ Cấu hình hệ thống

Các thông số vận hành có thể tùy chỉnh trong file [`config/config.yaml`](config/config.yaml):

```yaml
detection:
  model: "yolov8n.pt"
  confidence: 0.4
  device: "cuda"          # "cuda" hoặc "cpu"

attributes:
  input_size: [224, 112]  # Tỷ lệ crop người đứng (2:1)
  batch_size: 16
  threshold:
    gender: 0.50
    hat: 0.62
    glasses: 0.50
    backpack: 0.50

matching:
  default_threshold: 0.7  # Ngưỡng điểm khớp mặc định
```

---

## 📜 License

Dự án được phân phối theo giấy phép MIT. Xem thêm tại [LICENSE](LICENSE).