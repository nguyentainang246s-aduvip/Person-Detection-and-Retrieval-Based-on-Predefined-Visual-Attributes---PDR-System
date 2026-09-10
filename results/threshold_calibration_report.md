# 🎚️ BÁO CÁO HIỆU CHUẨN NGƯỠNG THUỘC TÍNH PAR (ƯU TIÊN 3)
> Tối ưu hóa điểm cắt quyết định (Decision Threshold Calibration) nhằm tối đa hóa F1-Score và giảm thiểu False Positives.

---

## 1. BẢNG NGƯỠNG TỐI ƯU HÓA (F1-OPTIMAL THRESHOLDS)

| Thuộc Tính PAR | Ngưỡng Mặc Định Cũ | Ngưỡng Tối Ưu (F1-Max) | Precision (Độ chính xác) | Recall (Độ bao phủ) | F1-Score Đạt Được | Ghi Chú Kỹ Thuật |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Giới tính (Female)** | 0.50 | **0.44** | 92.2% | 98.8% | **95.4%** | Cân bằng phân phối Nam/Nữ |
| **Đội mũ (Hat)** | 0.50 | **0.54** | 90.2% | 94.9% | **92.5%** | Nâng cao để triệt tiêu nhầm lẫn tóc đen/búi tóc |
| **Đeo kính (Glasses)** | 0.50 | **0.58** | 96.7% | 89.4% | **92.9%** | Nhận diện kính mắt cự ly gần & xa |
| **Đeo balo (Backpack)** | 0.50 | **0.56** | 81.2% | 90.7% | **85.7%** | Bắt nhạy balo và túi đeo chéo |

---

## 2. KHUYẾN NGHỊ THEO TỪNG MÔI TRƯỜNG GIÁM SÁT THỰC TẾ

1. **Camera Ngoài Trời / CCTV Góc Cao (Ánh sáng gắt, bóng râm):**
   - Đặt `hat` = **0.65** (Bóng râm trên đỉnh đầu hay tạo ảo giác đội mũ).
   - Đặt `gender` = **0.50**, `backpack` = **0.50**.
2. **Trong Nhà / Văn Phòng (Ánh sáng đồng đều):**
   - Đặt `hat` = **0.60**, `glasses` = **0.45**, `backpack` = **0.45**.
3. **Môi Trường Thiếu Sáng / Ngược Sáng (Khuất bóng, ban đêm):**
   - Nâng `hat` = **0.70**, `glasses` = **0.55** để lọc sạch nhiễu hạt ISO của camera.
