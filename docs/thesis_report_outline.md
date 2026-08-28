# ĐỀ CƯƠNG CHI TIẾT BÁO CÁO ĐỒ ÁN TỐT NGHIỆP
## Đề tài: Xây dựng giải pháp phát hiện / tìm người dựa trên đặc điểm nhận dạng cho trước
### (Person Detection and Retrieval Based on Predefined Visual Attributes)

---

## 📑 MỤC LỤC TỔNG THỂ BÁO CÁO (5 CHƯƠNG CHUẨN ĐẠI HỌC)

### LỜI CẢM ƠN
### TÓM TẮT ĐỒ ÁN (TIẾNG VIỆT & TIẾNG ANH - ABSTRACT)
### DANH MỤC THUẬT NGỮ VÀ KÝ HIỆU VIẾT TẮT
### DANH MỤC BẢNG BIỂU & HÌNH VẼ

---

## CHƯƠNG 1: TỔNG QUAN VÀ ĐẶT VẤN ĐỀ
* **1.1. Bối cảnh thực tiễn**:
  * Sự phát triển mạnh mẽ của hệ thống camera giám sát an ninh (CCTV) tại đô thị thông minh, sân bay, nhà ga, trung tâm thương mại.
  * Thách thức của việc tìm kiếm đối tượng người mất tích, kẻ gian hoặc tìm người theo mô tả nhân chứng trong hàng ngàn giờ video thủ công.
* **1.2. Mục tiêu nghiên cứu của Đề tài**:
  * Tự động hóa quá trình tìm kiếm người trong video dựa trên tập thuộc tính ngoại hình chọn trước (Giới tính, Màu áo, Màu quần, Phụ kiện balo, mũ, kính).
  * Đạt tốc độ xử lý thời gian thực (Real-time).
* **1.3. Đối tượng và Phạm vi nghiên cứu**:
  * Đối tượng: Video giám sát, hình ảnh người đi bộ (Pedestrians).
  * Phạm vi: Không gian màu HSV, mô hình YOLOv8, ByteTrack, ResNet50, dataset PA-100K.
* **1.4. Đóng góp của Đồ án**:
  * Xây dựng trọn vẹn pipeline end-to-end từ tiền xử lý, detection, tracking, attribute recognition đến retrieval scoring.
  * Thiết kế giao diện web trực quan Streamlit và cơ sở dữ liệu SQLite lưu trữ lịch sử.
* **1.5. Bố cục của Đồ án**.

---

## CHƯƠNG 2: CƠ SỞ LÝ THUYẾT VÀ CÁC CÔNG NGHỆ LIÊN QUAN
* **2.1. Bài toán Phát hiện đối tượng (Object Detection) & Kiến trúc YOLO**:
  * Khái niệm mạng nơ-ron tích chập (CNN).
  * So sánh Two-stage (R-CNN) vs One-stage (YOLO).
  * Kiến trúc YOLOv8: Backbone CSPDarknet, Neck PAN-FPN, Anchor-free Head và hàm mất mát (CIoU, DFL).
* **2.2. Bài toán Theo dõi đa đối tượng (Multi-Object Tracking - MOT) & ByteTrack**:
  * Bộ lọc Kalman (Kalman Filter) trong ước lượng chuyển động.
  * Thuật toán Hungarian trong ghép cặp ma trận IoU.
  * Điểm đột phá của ByteTrack: Cơ chế liên kết 2 bước (Two-stage Association) cứu các detection có confidence thấp bị che khuất.
* **2.3. Không gian màu và Phương pháp Phân tích màu sắc**:
  * So sánh không gian màu RGB vs HSV.
  * Thuật toán gom cụm K-Means (K-Means Clustering) để lọc nhiễu và trích xuất màu chủ đạo trang phục.
* **2.4. Bài toán Nhận dạng thuộc tính người (Person Attribute Recognition - PAR)**:
  * Khái niệm Multi-label Classification (Phân loại đa nhãn).
  * Kiến trúc mạng ResNet50 (Residual Connections giải quyết hiện tượng suy biến gradient - Vanishing Gradient).
  * Hàm mất mát Binary Cross-Entropy (BCELoss).
* **2.5. Tập dữ liệu chuẩn PA-100K**:
  * Đặc điểm 100.000 ảnh và cấu trúc nhãn 26 thuộc tính.

---

## CHƯƠNG 3: THIẾT KẾ VÀ XÂY DỰNG KIẾN TRÚC HỆ THỐNG
* **3.1. Sơ đồ kiến trúc tổng thể hệ thống (System Architecture)**.
* **3.2. Thiết kế Module Phát hiện người (Person Detection Module)**.
* **3.3. Thiết kế Module Theo dõi đa đối tượng (Tracking Module)**.
* **3.4. Thiết kế Module Trích xuất vùng đặc trưng & Phân tích màu sắc**:
  * Phân đoạn tỷ lệ cơ thể ($15\%-55\%$ thân trên, $55\%-90\%$ thân dưới).
* **3.5. Thiết kế Module Nhận dạng thuộc tính (PAR Module)**:
  * Cấu trúc ResNet50 Backbone + FC Classification Head + Sigmoid.
* **3.6. Thiết kế Bộ máy So khớp và Tính điểm (Matching Engine)**:
  * Công thức tính điểm có trọng số và cơ chế xử lý thuộc tính `"Any"`.
* **3.7. Thiết kế Cơ sở dữ liệu SQLite**:
  * Lược đồ ERD bảng `queries` và `search_results`.
* **3.8. Thiết kế Giao diện người dùng Streamlit (UI/UX)**.

---

## CHƯƠNG 4: CÀI ĐẶT THỰC NGHIỆM VÀ ĐÁNH GIÁ KẾT QUẢ
* **4.1. Môi trường thực nghiệm (Phần cứng, Phần mềm, Thư viện)**.
* **4.2. Quá trình huấn luyện mô hình PAR trên Google Colab**:
  * Thiết lập Hyperparameters (Learning rate, Batch size, Optimizer, StepLR).
  * Đồ thị đường cong hàm mất mát (Loss Curve) và Mean Accuracy (mA) qua các Epoch.
* **4.3. Đánh giá độ chính xác từng module**:
  * YOLOv8 mAP@0.5 trên COCO person class.
  * Độ chính xác nhận dạng từng thuộc tính: Giới tính, Mũ, Kính, Balo, Màu áo, Màu quần.
* **4.4. Đo lường hiệu năng thời gian thực (Latency & FPS Benchmark)**:
  * Bảng phân tích thời gian trễ từng module trên CPU và GPU.
* **4.5. Đánh giá kết quả tìm kiếm trên các kịch bản video thực tế**:
  * Kịch bản 1: Người đi một mình, hình ảnh rõ nét.
  * Kịch bản 2: Người bị che khuất một phần (Occlusion).
  * Kịch bản 3: Video có nhiều người qua lại cùng lúc.

---

## CHƯƠNG 5: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN
* **5.1. Kết luận những kết quả đã đạt được**:
  * Xây dựng thành công hệ thống hoàn chỉnh chạy mượt mà từ video đầu vào đến giao diện web và cơ sở dữ liệu.
* **5.2. Những hạn chế còn tồn tại**:
  * Ánh sáng ban đêm quá yếu làm sai lệch màu sắc HSV.
  * Người bị che khuất quá $60\%$ trong thời gian dài có thể bị gán ID mới.
* **5.3. Hướng phát triển trong tương lai**:
  * Tích hợp Vision-Language Model (như CLIP) để hỗ trợ tìm kiếm bằng câu mô tả tự nhiên (Free-text prompt).
  * Bổ sung Re-ID Embedding để nhận diện lại người khi rời khỏi góc quay camera.

---

## TÀI LIỆU THAM KHẢO
## PHỤ LỤC (SOURCE CODE VÀ HƯỚNG DẪN CÀI ĐẶT)
