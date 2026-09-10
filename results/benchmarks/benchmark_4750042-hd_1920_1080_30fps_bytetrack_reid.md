# Kết Quả Thực Nghiệm — 4750042-hd_1920_1080_30fps

- **Video:** `data/test_videos/4750042-hd_1920_1080_30fps.mp4` (393 frames)
- **Tracker:** `bytetrack_reid`
- **Tốc độ:** 6.1 FPS (164.0 ms/frame)

### 1. Chỉ số Tracking (motmetrics)

| Chỉ số | Giá trị | Ý nghĩa |
| :--- | :--- | :--- |
| **MOTA** | **90.16%** | Độ chính xác tracking tổng quát |
| **IDF1** | **90.17%** | Khả năng duy trì đúng ID (Trọng tâm Re-ID) |
| **ID-Switches** | **10 lần** | Số lần bị nhảy nhầm ID |
| **False Positives** | 298 | Số phát hiện nhầm (báo ảo) |
| **Misses** | 6 | Số đối tượng bị bỏ sót |

### 2. Chỉ số Nhận dạng Thuộc tính (PAR)

- **Mean Accuracy (mA chuẩn PAR):** **52.56%**

| Thuộc tính | mA (%) | Precision (%) | Recall (%) | F1 Score (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Giới tính (Gender)** | 62.9 | 79.4 | 68.3 | 73.4 |
| **Đội mũ (Hat)** | 47.5 | 0.0 | 0.0 | 0.0 |
| **Kính mắt (Glasses)** | 49.8 | 0.0 | 0.0 | 0.0 |
| **Balo/Túi (Backpack)** | 50.0 | 0.0 | 0.0 | 0.0 |

### 3. Chỉ số Nhận dạng Màu sắc (Color)

- **Màu áo (Upper Color Acc):** **64.21%**
- **Màu quần (Lower Color Acc):** **40.50%**
- **Độ chính xác màu trung bình:** **52.36%**
