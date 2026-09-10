# Bảng Số Liệu Thực Nghiệm Màu Sắc (Ablation Study — Giai Đoạn 2)

| Phương pháp | Accuracy (Thường) | Accuracy (Ngược sáng / Bóng đổ) | Accuracy (Họa tiết) | Độ trễ (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **HSV K-Means (Thuần)** | **51.80%** | **25.40%** | **47.00%** | 0.90 ms |
| **HSV + Delta-E Fallback** | **51.80%** | **24.40%** | **47.00%** | 0.75 ms |
| **Learned Color Head (Deep Learning)** | **44.20%** | **38.60%** | **39.20%** | 12.13 ms |

### Nhận xét & Phân tích Trade-off (Tính toán tự động từ số liệu thực nghiệm):
1. **Độ chính xác (Accuracy):** Ở điều kiện thường trên tập test độc lập, Learned Color Head đạt **44.20%** (chênh lệch -7.60% so với HSV K-Means). Trong điều kiện ánh sáng khó, mô hình kháng bóng đổ / ngược sáng đạt **38.60%** (chênh lệch +13.20%) và khi gặp họa tiết đạt **39.20%** (chênh lệch -7.80%).
2. **Độ trễ (Latency):** Phương pháp HSV K-Means có độ trễ **0.90 ms**, trong khi Learned Color Head mất **12.13 ms** (chênh lệch +11.22 ms, hoàn toàn đáp ứng ngưỡng thời gian thực).
