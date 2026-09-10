# Kết Quả Thực Nghiệm — 4750061-hd_1920_1080_30fps

- **Video:** `data/test_videos/4750061-hd_1920_1080_30fps.mp4` (769 frames)
- **Tracker:** `bytetrack_reid`
- **Tốc độ:** 5.7 FPS (174.5 ms/frame)

### 1. Chỉ số Tracking (motmetrics)

| Chỉ số | Giá trị | Ý nghĩa |
| :--- | :--- | :--- |
| **MOTA** | **93.97%** | Độ chính xác tracking tổng quát |
| **IDF1** | **96.75%** | Khả năng duy trì đúng ID (Trọng tâm Re-ID) |
| **ID-Switches** | **6 lần** | Số lần bị nhảy nhầm ID |
| **False Positives** | 429 | Số phát hiện nhầm (báo ảo) |
| **Misses** | 1 | Số đối tượng bị bỏ sót |

### 2. Chỉ số Nhận dạng Thuộc tính (PAR)

- **Mean Accuracy (mA chuẩn PAR):** **48.85%**

| Thuộc tính | mA (%) | Precision (%) | Recall (%) | F1 Score (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Giới tính (Gender)** | 49.3 | 90.6 | 59.9 | 72.2 |
| **Đội mũ (Hat)** | 47.3 | 0.0 | 0.0 | 0.0 |
| **Kính mắt (Glasses)** | 48.8 | 0.0 | 0.0 | 0.0 |
| **Balo/Túi (Backpack)** | 50.0 | 0.0 | 0.0 | 0.0 |

### 3. Chỉ số Nhận dạng Màu sắc (Color)

- **Màu áo (Upper Color Acc):** **62.54%**
- **Màu quần (Lower Color Acc):** **54.86%**
- **Độ chính xác màu trung bình:** **58.70%**
