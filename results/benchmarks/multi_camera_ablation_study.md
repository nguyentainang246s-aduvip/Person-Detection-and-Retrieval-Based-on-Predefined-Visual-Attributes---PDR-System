# Báo Cáo Thực Nghiệm Multi-Camera Ingestion & Session Isolation (Giai Đoạn 5)

### Bảng 1: So sánh Hiệu năng Xử lý Đa luồng Multi-Camera

| Kịch bản Camera | FPS trung bình / Cam | Tổng thông lượng hệ thống | Độ trễ bình quân | Tỉ lệ rớt frame | Tài nguyên (CPU / RAM) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1 Camera đồng thời** | **29.81 FPS** | **29.81 FPS** | 28.1 ms | 0.0% | 91.2% CPU / 496.3 MB |
| **2 Camera đồng thời** | **27.19 FPS** | **54.38 FPS** | 31.28 ms | 0.0% | 99.1% CPU / 519.1 MB |
| **4 Camera đồng thời** | **24.57 FPS** | **98.27 FPS** | 35.2 ms | 0.0% | 99.8% CPU / 574.7 MB |

### Cấu hình Phần cứng Thử nghiệm (Hardware Environment):
- **CPU:** 4 Cores vật lý / 8 Threads logic (x86_64, Windows 11)
- **RAM:** 15.84 GB DDR4
- **GPU:** NVIDIA GeForce GTX 1650 (4.00 GB VRAM, GDDR6)
- **Kiến trúc đa luồng:** `CameraStreamReader` (Hàng đợi vòng `Queue(maxsize=20)`) + `ThreadPoolExecutor` worker pool.

### Phân tích Tranh chấp Tài nguyên (Resource Contention Analysis):
1. **Suy giảm FPS/Camera:** Khi tải tăng từ 1 lên 4 camera, FPS trung bình mỗi camera giảm từ **29.81 FPS** xuống **24.57 FPS** (giảm 17.6%), tổng thông lượng đạt **98.27 FPS**.
2. **Nguyên nhân vật lý:** Hiện tượng suy giảm xảy ra do tranh chấp bộ nhớ (memory bus bandwidth), cache L2/L3 contention, và chuyển đổi ngữ cảnh giữa 4 luồng camera chạy đồng thời trên 4 core CPU vật lý.
3. **Cơ chế Backpressure Handling:** Khi worker quá tải ở kịch bản 4 camera, hàng đợi `CameraStreamReader` kích hoạt cơ chế drop oldest frame để loại bỏ độ trễ tích lũy, bảo đảm hiển thị luôn là thời gian thực (Zero-Lag Display).
