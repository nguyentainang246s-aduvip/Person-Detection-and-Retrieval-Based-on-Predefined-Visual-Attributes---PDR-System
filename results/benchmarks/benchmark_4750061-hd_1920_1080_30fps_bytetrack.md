# Kết Quả Thực Nghiệm — 4750061-hd_1920_1080_30fps

- **Video:** `data/test_videos/4750061-hd_1920_1080_30fps.mp4` (769 frames)
- **Tracker:** `bytetrack`
- **Tốc độ:** 9.1 FPS (110.0 ms/frame)

### 1. Chỉ số Tracking (motmetrics)

| Chỉ số | Giá trị | Ý nghĩa |
| :--- | :--- | :--- |
| **MOTA** | **91.96%** | Độ chính xác tracking tổng quát |
| **IDF1** | **92.40%** | Khả năng duy trì đúng ID (Trọng tâm Re-ID) |
| **ID-Switches** | **18 lần** | Số lần bị nhảy nhầm ID |
| **False Positives** | 560 | Số phát hiện nhầm (báo ảo) |
| **Misses** | 3 | Số đối tượng bị bỏ sót |

### 2. Chỉ số Nhận dạng Thuộc tính (PAR)

- **Mean Accuracy (mA chuẩn PAR):** **48.90%**

| Thuộc tính | mA (%) | Precision (%) | Recall (%) | F1 Score (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Giới tính (Gender)** | 50.2 | 90.9 | 61.0 | 73.0 |
| **Đội mũ (Hat)** | 47.2 | 0.0 | 0.0 | 0.0 |
| **Kính mắt (Glasses)** | 48.2 | 0.0 | 0.0 | 0.0 |
| **Balo/Túi (Backpack)** | 50.0 | 0.0 | 0.0 | 0.0 |

### 3. Chỉ số Nhận dạng Màu sắc (Color)

- **Màu áo (Upper Color Acc):** **63.71%**
- **Màu quần (Lower Color Acc):** **55.11%**
- **Độ chính xác màu trung bình:** **59.41%**
