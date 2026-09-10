# Kết Quả Thực Nghiệm — 4750042-hd_1920_1080_30fps

- **Video:** `data/test_videos/4750042-hd_1920_1080_30fps.mp4` (393 frames)
- **Tracker:** `botsort_reid`
- **Tốc độ:** 4.6 FPS (218.1 ms/frame)

### 1. Chỉ số Tracking (motmetrics)

| Chỉ số | Giá trị | Ý nghĩa |
| :--- | :--- | :--- |
| **MOTA** | **89.50%** | Độ chính xác tracking tổng quát |
| **IDF1** | **84.56%** | Khả năng duy trì đúng ID (Trọng tâm Re-ID) |
| **ID-Switches** | **19 lần** | Số lần bị nhảy nhầm ID |
| **False Positives** | 310 | Số phát hiện nhầm (báo ảo) |
| **Misses** | 6 | Số đối tượng bị bỏ sót |

### 2. Chỉ số Nhận dạng Thuộc tính (PAR)

- **Mean Accuracy (mA chuẩn PAR):** **52.47%**

| Thuộc tính | mA (%) | Precision (%) | Recall (%) | F1 Score (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Giới tính (Gender)** | 62.0 | 78.9 | 66.8 | 72.4 |
| **Đội mũ (Hat)** | 48.3 | 0.0 | 0.0 | 0.0 |
| **Kính mắt (Glasses)** | 49.6 | 0.0 | 0.0 | 0.0 |
| **Balo/Túi (Backpack)** | 50.0 | 0.0 | 0.0 | 0.0 |

### 3. Chỉ số Nhận dạng Màu sắc (Color)

- **Màu áo (Upper Color Acc):** **65.38%**
- **Màu quần (Lower Color Acc):** **39.84%**
- **Độ chính xác màu trung bình:** **52.61%**
