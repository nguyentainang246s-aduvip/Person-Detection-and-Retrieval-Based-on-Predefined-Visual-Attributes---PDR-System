# 🔍 Person Detection & Retrieval Based on Predefined Visual Attributes

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1+cu121-orange?logo=pytorch)
![YOLOv8](https://img.shields.io/badge/YOLO-v8n-green?logo=yolo)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_UI-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**Đồ án tốt nghiệp** – Xây dựng giải pháp phát hiện / tìm người dựa trên đặc điểm nhận dạng cho trước

*Person Detection and Retrieval Based on Predefined Visual Attributes*

</div>

---

## 🎯 Giới thiệu

Hệ thống cho phép người dùng **mô tả đặc điểm ngoại hình** của người cần tìm (giới tính, màu áo, màu quần, balo, mũ, kính) và **tự động tìm kiếm** người đó trong video giám sát theo thời gian thực.

**Ví dụ:** Tìm *"Nam, áo đen, đeo balo"* → hệ thống phát hiện, theo dõi và highlight đúng người trong video.

---

## ✨ Tính năng chính

| Tính năng | Mô tả | Công nghệ |
|---|---|---|
| 🎯 Person Detection | Phát hiện người trong video | YOLOv8n (COCO pretrained) |
| 🔄 Multi-Object Tracking | Gán Track ID ổn định liên frame | ByteTrack |
| 👔 Color Detection | Nhận diện màu áo + màu quần | HSV + Trimmed Median |
| 🧠 Attribute Recognition | Giới tính, Mũ, Kính, Balo | ResNet50 (PA-100K, mA=89.33%) |
| ⏱️ Temporal Smoothing | Giảm rung giật nhãn khi tracking | Exponential Moving Average |
| 🎯 Matching Engine | Chấm điểm tương đồng có trọng số | Weighted Score Algorithm |
| 🌐 Web UI | Giao diện demo trực quan | Streamlit (100% Tiếng Việt) |
| 🗄️ Database | Lưu lịch sử tìm kiếm | SQLite |

---

## 📊 Kết quả Đánh giá

### PAR Model Accuracy (PA-100K Dataset, 10,000 test images)

| Thuộc tính | Accuracy |
|---|---|
| Giới tính (Gender) | **85.20%** |
| Đội Mũ (Hat) | **84.70%** |
| Đeo Kính (Glasses) | **91.00%** |
| Đeo Balo (Backpack) | **96.70%** |
| ⭐ **Mean Accuracy (mA)** | **89.33%** |

### FPS Benchmark (i5-10300H)

| Module | CPU (i5-10300H) | GPU (GTX 1650) |
|---|---|---|
| YOLOv8n Detection | 16.9 FPS | 76.3 FPS |
| ByteTrack Tracking | 16.6 FPS | 72.2 FPS |
| ResNet50 PAR | 23.3 FPS | 87.3 FPS |
| HSV Color Detector | 9,754 FPS | 9,983 FPS |
| **⭐ Full Pipeline** | **~16 FPS** | **~70 FPS** |

---

## 🏗️ Kiến trúc Pipeline

```
Video / Camera Input
        ↓
YOLOv8n Person Detection   → Phát hiện bounding box người
        ↓
ByteTrack MOT Tracking     → Gán Track ID ổn định liên frame
        ↓
Person Crop ROI Extraction → Trích xuất vùng ảnh mỗi người
        ↓  ┌────────────────────────────────────┐
        ↓  │  HSV+Trimmed Median Color (~0.1ms) │
        ↓  │  ResNet50 PAR Classifier (~11ms)   │
        ↓  │  EMA Temporal Smoothing (α=0.35)   │
        ↓  └────────────────────────────────────┘
        ↓
Weighted Attribute Matching → Tính điểm % khớp với query
        ↓
Person Retrieval Result    → Highlight Target / Others
        ↓
Streamlit Display + SQLite Save
```

---

## 📁 Cấu trúc thư mục

```
person-retrieval/
├── app.py                    ← Entry point: Streamlit Web App
├── config/config.yaml        ← Cấu hình hệ thống
├── requirements.txt          ← Dependencies
│
├── src/
│   ├── detection/detector.py ← YOLOv8n Person Detector
│   ├── tracking/tracker.py   ← ByteTrack Tracker
│   ├── attributes/
│   │   ├── color_detector.py ← HSV + Trimmed Median Color Classifier
│   │   └── par_model.py      ← ResNet50 PAR (Gender/Hat/Glasses/Backpack)
│   ├── retrieval/
│   │   ├── matcher.py        ← Weighted Attribute Matching Engine
│   │   └── pipeline.py       ← End-to-End Pipeline Orchestrator
│   ├── database/db.py        ← SQLite Search History Manager
│   └── utils/                ← Logger, Video Utils, Visualization
│
├── models/
│   ├── yolo/yolov8n.pt       ← YOLOv8n model (6.2 MB, COCO pretrained)
│   └── par/par_resnet50.pth  ← PAR model (94 MB, PA-100K fine-tuned)
│
├── training/
│   ├── train_par.py          ← Training script (Colab-ready, Colab T4)
│   └── dataset_pa100k.py     ← PA-100K DataLoader
│
├── scripts/
│   ├── evaluate_par.py       ← Evaluation script (mA, Precision, Recall)
│   ├── benchmark_fps.py      ← FPS Benchmark tool
│   ├── demo_pipeline.py      ← CLI demo
│   └── test_*.py             ← Module test scripts
│
├── docs/
│   ├── architecture.md       ← Chi tiết kiến trúc hệ thống
│   ├── defense_qa_guide.md   ← Bộ câu hỏi phản biện bảo vệ (12 câu)
│   └── thesis_report_outline.md ← Outline báo cáo đồ án
│
└── data/
    └── test_videos/          ← Video test mẫu
```

---

## 🚀 Cài đặt và Chạy

### 1. Yêu cầu hệ thống
- Python 3.10+
- NVIDIA GPU (khuyến nghị, GTX 1650+ / RTX series)
- CUDA 12.1 (nếu dùng GPU)
- RAM: 8 GB+, VRAM: 4 GB+

### 2. Cài đặt

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/person-retrieval.git
cd person-retrieval

# Tạo virtual environment
python -m venv venv
.\venv\Scripts\activate       # Windows
# source venv/bin/activate    # Linux/Mac

# Cài dependencies
pip install -r requirements.txt

# Cài PyTorch với CUDA (nếu có GPU NVIDIA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Cài PyTorch CPU only (nếu không có GPU)
pip install torch torchvision
```

### 3. Chạy Web Demo

```bash
streamlit run app.py
```

Mở trình duyệt tại: `http://localhost:8501`

### 4. Chạy qua CLI

```bash
# Tìm người Nam mặc áo đen
python scripts/demo_pipeline.py --gender Male --upper-color Black

# Tìm người đội mũ đeo balo
python scripts/demo_pipeline.py --hat --backpack
```

### 5. Benchmark FPS

```bash
python scripts/benchmark_fps.py
```

### 6. Đánh giá Model (cần dataset PA-100K)

```bash
# Hiển thị kết quả Colab training:
python scripts/evaluate_par.py --demo

# Đánh giá trên dataset thực:
python scripts/evaluate_par.py --data-root datasets/PA100K
```

---

## 🎓 Training PAR Model

PAR Model (ResNet50) được fine-tune trên [PA-100K dataset](https://github.com/xh-liu/HydraPlus-Net#pa-100k-dataset).

```bash
# Trên Google Colab T4 GPU:
python training/train_par.py
# Output: models/par/par_resnet50.pth
# Thời gian: ~45 phút (20 epochs, batch=32, T4 GPU)
```

**Kết quả training:**
- Val mA: **89.33%** (epoch 18/20)
- Training device: Colab T4 GPU
- Training samples: 90,000 | Val samples: 10,000

---

## 📱 Giao diện Web Demo

Web app gồm 4 tab:

| Tab | Chức năng |
|---|---|
| 🎯 Tìm kiếm Trực quan | Upload video → Chọn đặc điểm → Tìm kiếm real-time |
| 📜 Lịch sử Tìm kiếm | Xem lại các phiên tìm kiếm đã lưu trong SQLite DB |
| 📊 Đánh giá Mô hình | Bảng Accuracy, biểu đồ Plotly, FPS benchmark CPU vs GPU |
| ℹ️ Giới thiệu | Sơ đồ pipeline, mô tả thuật toán |

---

## 🔧 Cấu hình

Tất cả tham số được quản lý trong [`config/config.yaml`](config/config.yaml):

```yaml
detection:
  model: "yolov8n.pt"
  confidence: 0.4
  device: "cuda"          # Tự động dùng GPU nếu có

attributes:
  input_size: [224, 112]  # Tỷ lệ người đứng (2:1)
  threshold:
    gender: 0.50
    hat: 0.62             # Nâng cao tránh nhầm tóc đen

matching:
  default_threshold: 0.7  # Ngưỡng điểm khớp tối thiểu
```

---

## 📚 Tài liệu tham khảo

- [YOLOv8 - Ultralytics](https://github.com/ultralytics/ultralytics)
- [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864)
- [PA-100K Dataset - HydraPlus-Net](https://github.com/xh-liu/HydraPlus-Net#pa-100k-dataset)
- [ResNet: Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## 👤 Tác giả

**Nguyễn Đức Tài Năng**
- Email: nguyentainang246@gmail.com
- Đồ án tốt nghiệp ngành Điện tử Viễn thông / Kỹ thuật máy tính

---

## 📄 License

MIT License — xem file [LICENSE](LICENSE) để biết thêm chi tiết.
