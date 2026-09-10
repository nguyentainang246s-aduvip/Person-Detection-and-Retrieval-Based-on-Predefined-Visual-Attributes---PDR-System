# Kết Quả Thực Nghiệm — 4750042-hd_1920_1080_30fps

- **Video:** `data/test_videos/4750042-hd_1920_1080_30fps.mp4` (393 frames)
- **Tracker:** `bytetrack`
- **Tốc độ:** 7.6 FPS (131.1 ms/frame)

### 1. Chỉ số Tracking (motmetrics)

| Chỉ số | Giá trị | Ý nghĩa |
| :--- | :--- | :--- |
| **MOTA** | **86.93%** | Độ chính xác tracking tổng quát |
| **IDF1** | **87.50%** | Khả năng duy trì đúng ID (Trọng tâm Re-ID) |
| **ID-Switches** | **18 lần** | Số lần bị nhảy nhầm ID |
| **False Positives** | 398 | Số phát hiện nhầm (báo ảo) |
| **Misses** | 1 | Số đối tượng bị bỏ sót |

### 2. Chỉ số Nhận dạng Thuộc tính (PAR)

- **Mean Accuracy (mA chuẩn PAR):** **52.06%**

| Thuộc tính | mA (%) | Precision (%) | Recall (%) | F1 Score (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Giới tính (Gender)** | 61.8 | 78.6 | 68.0 | 72.9 |
| **Đội mũ (Hat)** | 47.3 | 0.0 | 0.0 | 0.0 |
| **Kính mắt (Glasses)** | 49.2 | 0.0 | 0.0 | 0.0 |
| **Balo/Túi (Backpack)** | 50.0 | 0.0 | 0.0 | 0.0 |

### 3. Chỉ số Nhận dạng Màu sắc (Color)

- **Màu áo (Upper Color Acc):** **64.55%**
- **Màu quần (Lower Color Acc):** **42.14%**
- **Độ chính xác màu trung bình:** **53.35%**
