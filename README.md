# 🔍 Person Detection & Retrieval System (PDR-System)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2Bcu121-orange?logo=pytorch)
![YOLOv8](https://img.shields.io/badge/YOLO-v8n-green?logo=yolo)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_UI-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**Hệ thống phát hiện và tìm kiếm người trong video dựa trên đặc điểm nhận dạng cho trước**

*Person Detection and Retrieval Based on Predefined Visual Attributes*

</div>

---

## 🎯 Giới thiệu

Đây là một project mã nguồn mở hỗ trợ **phát hiện và tìm kiếm người** trong video giám sát hoặc video tải lên dựa trên các thuộc tính ngoại hình được định trước (giới tính, màu áo, màu quần, phụ kiện như mũ, kính, balo).

Hệ thống kết hợp mô hình thị giác máy tính và học sâu (YOLOv8 + ByteTrack + ResNet50 PAR + Phân tích màu HSV) để xử lý và hiển thị kết quả theo thời gian thực qua giao diện web trực quan.

---

## ✨ Tính năng chính

- 🎯 **Phát hiện người (Person Detection):** Sử dụng YOLOv8n để phát hiện người trong từng khung hình.
- 🔄 **Theo dõi đối tượng (Tracking):** ByteTrack duy trì Track ID ổn định qua các khung hình, giảm thiểu mất dấu khi bị che khuất tạm thời.
- 👔 **Phân tích màu sắc (Color Recognition):** Trích xuất màu áo và màu quần trong không gian màu HSV kết hợp thuật toán lọc nhiễu Trimmed Median (xử lý ~0.1ms/người).
- 🧠 **Nhận diện thuộc tính (Attribute Recognition):** ResNet50 nhận dạng giới tính, mũ, kính, balo.
- ⏱️ **Làm mịn nhãn theo thời gian (Temporal EMA Smoothing):** Giảm rung giật nhãn phân loại qua chuỗi các khung hình liên tiếp.
- 🎛️ **Bộ máy so khớp linh hoạt (Weighted Matching Engine):** Chấm điểm độ tương đồng (%) theo trọng số, hỗ trợ tìm kiếm linh hoạt với các thuộc tính tùy chọn (`Any`).
- 🌐 **Giao diện Web tương tác (Streamlit):** Xem video trực tiếp, điều chỉnh bộ lọc, xem thống kê FPS và lưu lại lịch sử tìm kiếm vào SQLite.

---

## 🏗️ Luồng xử lý Pipeline

```
Video / Camera Input
        ↓
YOLOv8n Person Detection   → Phát hiện bounding box người
        ↓
ByteTrack MOT Tracking     → Gán Track ID ổn định qua các frame
        ↓
Person Crop ROI Extraction → Trích xuất vùng ảnh từng người
        ↓  ┌────────────────────────────────────────┐
        ↓  │  HSV + Trimmed Median Color (~0.1ms)   │
        ↓  │  ResNet50 PAR Classifier (~11ms on GPU)│
        ↓  │  Temporal EMA Smoothing (α=0.35)       │
        ↓  └────────────────────────────────────────┘
        ↓
Weighted Attribute Matching → Tính điểm tương đồng khớp với query
        ↓
Person Retrieval Result    → Highlight đối tượng thỏa mãn
        ↓
Streamlit Web UI + SQLite Database
```

---

## 📁 Cấu trúc thư mục

```
├── app.py                    # Giao diện Web Streamlit
├── config/config.yaml        # Cấu hình tham số hệ thống
├── requirements.txt          # Danh sách thư viện phụ thuộc
│
├── src/
│   ├── detection/detector.py # Module phát hiện người (YOLOv8)
│   ├── tracking/tracker.py   # Module theo dõi đối tượng (ByteTrack)
│   ├── attributes/
│   │   ├── color_detector.py # Phân tích màu áo / quần (HSV + Trimmed Median)
│   │   └── par_model.py      # Nhận diện thuộc tính (ResNet50 PAR)
│   ├── retrieval/
│   │   ├── matcher.py        # Bộ so khớp thuộc tính có trọng số
│   │   └── pipeline.py       # Pipeline tích hợp toàn bộ luồng xử lý
│   ├── database/db.py        # Quản lý lưu trữ SQLite
│   └── utils/                # Tiện ích logging, video, vẽ khung visualization
│
├── models/
│   ├── yolo/yolov8n.pt       # Trọng số YOLOv8n
│   └── par/par_resnet50.pth  # Trọng số ResNet50 PAR
│
├── training/
│   ├── train_par.py          # Script huấn luyện mô hình PAR
│   └── dataset_pa100k.py     # DataLoader chuẩn bị dữ liệu
│
├── scripts/
│   ├── evaluate_par.py       # Đánh giá độ chính xác mô hình
│   ├── benchmark_fps.py      # Đo kiểm hiệu năng / FPS
│   └── demo_pipeline.py      # Chạy demo bằng dòng lệnh CLI
│
└── data/
    └── test_videos/          # Video mẫu kiểm thử
```

---

## 🚀 Hướng dẫn cài đặt và sử dụng

### 1. Yêu cầu môi trường
- Python 3.10 trở lên
- Card đồ họa NVIDIA (khuyến nghị để đạt FPS cao) hoặc chạy trên CPU

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
# source venv/bin/activate

# 3. Cài đặt các gói phụ thuộc
pip install -r requirements.txt

# 4. Cài đặt PyTorch hỗ trợ GPU CUDA (nếu máy có GPU NVIDIA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Chạy giao diện Web

```bash
streamlit run app.py
```
Truy cập trình duyệt tại địa chỉ: `http://localhost:8501`

### 4. Chạy thử nghiệm bằng CLI

```bash
# Tìm người Nam mặc áo Đen
python scripts/demo_pipeline.py --gender Male --upper-color Black

# Tìm người đội Mũ và đeo Balo
python scripts/demo_pipeline.py --hat --backpack
```

### 5. Benchmark FPS

```bash
python scripts/benchmark_fps.py
```

---

## 🔧 Cấu hình hệ thống

Các thông số vận hành có thể tùy chỉnh trực tiếp trong file [`config/config.yaml`](config/config.yaml):

```yaml
detection:
  model: "yolov8n.pt"
  confidence: 0.4
  device: "cuda"          # "cuda" hoặc "cpu"

attributes:
  input_size: [224, 112]  # Tỷ lệ crop người đứng (2:1)
  threshold:
    gender: 0.50
    hat: 0.62
    glasses: 0.50
    backpack: 0.50

matching:
  default_threshold: 0.7  # Ngưỡng điểm khớp mặc định
```

---

## 📄 License

Project được phân phối theo giấy phép MIT. Xem thêm tại [LICENSE](LICENSE).
