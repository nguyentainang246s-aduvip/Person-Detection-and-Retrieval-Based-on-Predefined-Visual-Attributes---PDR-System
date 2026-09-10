# Báo Cáo Thực Nghiệm Tối Ưu Hóa Suy Luận (Giai Đoạn 4)

### Bảng 1: So sánh Độ trễ suy luận từng mô hình (Inference Latency)

| Module | PyTorch Eager (FP32) | ONNX Runtime (FP32) | ONNX Runtime (INT8) | Dung lượng mô hình | Hệ số tăng tốc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8n Detection** | 75.12 ms | 49.5 ms | - ms | 6.2 MB | **1.5x** |
| **PAR (MobileNetV3)** | 18.34 ms | 3.16 ms | 2.74 ms | 11.82 MB | **6.7x** |
| **DualColorNet (Color)** | 10.54 ms | 1.45 ms | 1.58 ms | 3.72 MB | **6.7x** |

### Bảng 2: Tốc độ xử lý toàn bộ hệ thống (End-to-End Pipeline FPS)

| Cấu hình Pipeline | FPS Xử lý | Độ trễ bình quân / Frame | Đánh giá khả năng Real-time |
| :--- | :---: | :---: | :--- |
| **Baseline (ResNet-50 + PyTorch Eager)** | **4.78 FPS** | 209.02 ms | Hệ thống ban đầu |
| **Giai đoạn 3 (MobileNetV3 + PyTorch Eager)** | **6.54 FPS** | 153.02 ms | Thay thế backbone nhẹ |
| **Giai đoạn 4 (ONNX Runtime + INT8 Quantized)** | **6.95 FPS** | 143.84 ms | Tối ưu hóa phần cứng toàn diện |

> **Chú thích quan trọng:** Bảng 2 đo tốc độ xử lý (FPS) thuần túy, không đánh giá độ chính xác trên nội dung ảnh thật — xem `par_ablation_study.md` và `color_ablation_study.md` cho số liệu độ chính xác.

### Đánh giá chất lượng sau lượng tử hóa (Quantization Loss):
- Độ tương đồng Cosine giữa PyTorch FP32 và ONNX INT8: **99.83%** (suy hao không đáng kể < 0.5%).
- Dung lượng mô hình PAR giảm từ **13.22 MB** xuống chỉ còn **11.82 MB** (giảm 10.6%), tối ưu hóa hoàn hảo cho lưu trữ trên thiết bị biên.
