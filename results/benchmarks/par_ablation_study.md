# Báo Cáo Thực Nghiệm PAR: Backbone & Loss Function (Giai Đoạn 3)

### Bảng 1: So sánh Kiến trúc Backbone

| Backbone | Tham số (Params) | Khối lượng tính toán (GFLOPs) | Balanced mA (Eval) | Latency đo thật (CPU) |
| :--- | :---: | :---: | :---: | :---: |
| **ResNet50 (Hiện tại)** | 24.56M | 4.10 GFLOPs | **45.76%** | **49.92 ms** |
| **MobileNetV3-Large** | 3.47M | 0.23 GFLOPs | **40.75%** | **12.37 ms** |
| **EfficientNet-B0** | 4.67M | 0.39 GFLOPs | **42.48%** | **19.43 ms** |

### Bảng 2: So sánh Hàm mất mát (Loss Function — Đánh giá trên tập Eval độc lập)

| Hàm mất mát (Loss Function) | Balanced mA Tổng | F1 (Glasses — thuộc tính hiếm nhất) | F1 (Hat) |
| :--- | :---: | :---: | :---: |
| **BCE + Pos_weight (Baseline)** | **49.47%** | **0.00%** | 0.00% |
| **Focal Loss (gamma=2.0)** | **49.75%** | **30.77%** | 0.00% |

> [!NOTE]
> **Phương pháp luận đánh giá & Kiểm soát Rò rỉ dữ liệu (Disjoint Identity / Data Leakage Control):**
> Thử nghiệm phân chia ngắt kết nối theo danh tính người (Disjoint Person ID Split): **39 người** (380 crops) cho tập train và **13 người** (139 crops) cho tập eval độc lập. Tất cả hình ảnh của cùng một người chỉ xuất hiện ở một trong hai tập, loại bỏ triệt để hiện tượng rò rỉ dữ liệu (data leakage) và học vẹt (overfitting) giữa các frame liên tiếp.
> 
> **Lưu ý về quy mô dữ liệu:** Tập eval gồm 139 mẫu từ 13 người chưa từng xuất hiện khi huấn luyện. Đây là kết quả thử nghiệm sơ bộ có đối chứng phương pháp luận chuẩn xác; để gia tăng tính khái quát hóa và độ tin cậy cho các thuộc tính hiếm (kính, mũ), cần tiếp tục mở rộng quy mô dữ liệu với các video còn lại hoặc kết hợp bộ benchmark quy mô lớn MSP60K.

### Kết luận nghiên cứu (Tính toán tự động từ số liệu thực nghiệm):
1. **MobileNetV3-Large** giảm **85.9%** tham số (3.47M vs 24.56M) và giảm **94.4%** GFLOPs (0.23 vs 4.10) so với ResNet50, tốc độ suy luận nhanh hơn **4.04x** (12.37 ms vs 49.92 ms).
2. **So sánh Hàm mất mát (Loss Function)**: Sau khi loại bỏ rò rỉ dữ liệu, Focal Loss đạt Balanced mA **49.75%** (so với BCE 49.47%, chênh lệch +0.28%). Trên các thuộc tính hiếm, F1(Glasses) đạt **30.77%** (chênh lệch +30.77%) và F1(Hat) đạt **0.00%** (chênh lệch +0.00%).
