# BỘ CÂU HỎI PHẢN BIỆN VÀ TRẢ LỜI MẪU BẢO VỆ ĐỒ ÁN TỐT NGHIỆP
## Đề tài: Xây dựng giải pháp phát hiện / tìm người dựa trên đặc điểm nhận dạng cho trước

---

### CÂU 1: Tại sao em lại chọn mô hình YOLOv8 thay vì các mô hình khác như Faster R-CNN hay SSD?
* **Trả lời chuẩn**:
  > "Dạ thưa Thầy/Cô, Faster R-CNN là mô hình Two-stage, tuy có độ chính xác cao nhưng tốc độ chỉ đạt khoảng 7-12 FPS, không đáp ứng được yêu cầu xử lý video giám sát thời gian thực. Trong khi đó, YOLOv8 là mô hình One-stage tiên tiến với kiến trúc Anchor-free Head và tối ưu hóa CSPDarknet, đạt được sự cân bằng xuất sắc giữa độ chính xác (mAP cao) và tốc độ (trên 30-50 FPS trên GPU phổ thông), giúp hệ thống của em có đủ thời gian trích xuất tiếp các thuộc tính ngoại hình trong cùng 1 frame."

---

### CÂU 2: Thuật toán ByteTrack hoạt động như thế nào và tại sao nó tốt hơn DeepSORT?
* **Trả lời chuẩn**:
  > "Dạ thưa Thầy/Cô, điểm yếu lớn nhất của DeepSORT là phụ thuộc vào mạng trích xuất đặc trưng Re-ID nặng nề và tự động loại bỏ các detection có điểm confidence thấp. ByteTrack giải quyết triệt để vấn đề này bằng **Cơ chế liên kết 2 bước (Two-stage Association)**:
  > 1. Bước 1 ghép nối các detection điểm cao ($conf \ge 0.5$).
  > 2. Bước 2 ghép nối các detection điểm thấp ($0.1 \le conf < 0.5$) với các track chưa được gán.
  > Nhờ đó, khi người đi qua vùng bị che khuất hoặc bóng râm khiến điểm YOLO giảm, ByteTrack vẫn giữ nguyên được Track ID mà không cần mạng Re-ID cồng kềnh, đạt tốc độ cực nhanh."

---

### CÂU 3: Tại sao nhóm không dùng mạng Deep Learning để nhận diện màu áo mà lại dùng HSV + Trimmed Median?
* **Trả lời chuẩn**:
  > "Dạ thưa Thầy/Cô, trong các tập dữ liệu chuẩn cho PAR như PA-100K, nhãn màu sắc thường không đầy đủ hoặc bị bias theo từng bối cảnh ánh sáng. Em sử dụng không gian màu **HSV** giúp tách biệt kênh độ sáng (Value) khỏi sắc độ màu thực (Hue), kết hợp cùng thuật toán **Trimmed Median** — lấy giá trị trung vị của vùng trung tâm pixel sau khi loại bỏ 20% viền biên. Giải pháp này chạy chỉ **0.1ms mỗi người** (so với K-Means là 1.7 giây cho 26 người), không tốn GPU và cho độ chính xác tốt trong điều kiện ánh sáng giữ ngưyên kiến trúc ROI thuyết phục."

---

### CÂU 4: Multi-label Classification trong nhận diện thuộc tính khác gì so với Image Classification thông thường?
* **Trả lời chuẩn**:
  > "Dạ thưa Thầy/Cô, trong Multi-class Classification thông thường (như phân loại 1000 lớp ImageNet), mỗi bức ảnh chỉ thuộc về 1 nhãn duy nhất và ta dùng hàm **Softmax** ($\sum p_i = 1$). Trong bài toán của em là **Multi-label Classification**, một người có thể cùng lúc vừa là Nữ, vừa Đội mũ, vừa Đeo balo. Các thuộc tính này độc lập với nhau nên lớp cuối cùng phải dùng hàm **Sigmoid** cho từng node đầu ra và tối ưu bằng hàm mất mát **Binary Cross-Entropy (BCELoss)**."

---

### CÂU 5: Công thức tính điểm tương đồng (Matching Score) của hệ thống được tính như thế nào?
* **Trả lời chuẩn**:
  > "Dạ thưa Thầy/Cô, em sử dụng công thức tính điểm tương đồng có trọng số:
  > $$\text{Score} = \frac{\sum (\text{Trọng số thuộc tính khớp})}{\sum (\text{Trọng số thuộc tính người dùng yêu cầu})} \times 100\%$$
  > Điểm đặc biệt là các thuộc tính người dùng chọn `'Any'` sẽ không bị tính vào mẫu số, đảm bảo người dùng không bị phạt điểm khi không quan tâm đến thuộc tính đó. Thuộc tính quan trọng như màu áo được gán trọng số $1.5$, các thuộc tính khác là $1.0$."

---

### CÂU 6: Làm thế nào hệ thống đảm bảo tốc độ thời gian thực (Real-time) khi vừa chạy YOLO, vừa Tracking, vừa chạy ResNet50?
* **Trả lời chuẩn**:
  > "Dạ thưa Thầy/Cô, em áp dụng 3 kỹ thuật song song:
  > 1. **Track Memory Caching**: ResNet50 chỉ chạy mỗi 5 frames cho mỗi Track ID (mỗi người), giảm tải ResNet50 xuống 80%.
  > 2. **Computational Load Budgeting**: Mỗi frame chỉ chạy CNN nặng cho tối đa 2 Track mới, những người còn lại dung kết quả cache. Track ID ổn định (chạy qua 3+ lần) chỉ cập nhật lại sau 20 frames.
  > 3. **Fast Color Extraction**: Thuật toán Trimmed Median thay vì K-Means giúp trích xuất màu chỉ 0.1ms/người (nhanh hơn 400 lần K-Means).
  > Kết quả benchmark thực tế: YOLOv8n ~59ms, ResNet50 ~43ms, Full Pipeline ~63ms = 16 FPS trên CPU (i5-10300H). Trên GPU GTX 1650: ước tính 20–30 FPS."

---

### CÂU 7: Nếu đối tượng bị che khuất hoàn toàn trong 10 giây rồi quay lại, hệ thống có nhận ra không?
* **Trả lời chuẩn**:
  > "Dạ thưa Thầy/Cô, nếu người ra khỏi khung hình quá thời gian lưu giữ của Kalman Filter (thường là 30 frames $\approx 1$ giây), ByteTrack sẽ gán một Track ID mới. Tuy nhiên, nhờ cơ chế tìm kiếm theo đặc điểm ngoại hình (Visual Attributes), ngay khi người đó xuất hiện trở lại, hệ thống sẽ phân tích lại màu sắc/thuộc tính và **vẫn tiếp tục phát hiện và highlight người đó là Target Found** dựa trên độ khớp đặc điểm, đảm bảo không bỏ sót mục tiêu."

---

### CÂU 8: Tại sao em lại chọn kích thước ảnh crop là 224x112 thay vì 224x224?
* **Trả lời chuẩn**:
  > "Dạ thưa Thầy/Cô, tỷ lệ cơ thể người chuẩn khi đứng là hình chữ nhật đứng (tỷ lệ xấp xỉ 2:1). Nếu ép ảnh về hình vuông 224x224, tỷ lệ người sẽ bị bóp méo theo chiều ngang, làm biến dạng đặc trưng hình học của balo, mũ hoặc dáng người. Kích thước $224 \times 112$ vừa giữ nguyên tỷ lệ dáng người tự nhiên, vừa giảm được $50\%$ số lượng pixel đầu vào giúp tăng tốc độ mạng ResNet50."

---

### CÂU 9: Hệ thống giải quyết bài toán quyền riêng tư (Privacy) như thế nào?
* **Trả lời chuẩn**:
  > "Dạ thưa Thầy/Cô, hệ thống của em chỉ phân tích **đặc điểm ngoại hình bề ngoài** (quần áo, giới tính nhìn từ xa, balo) để khoanh vùng đối tượng trong giám sát an ninh, **hoàn toàn không trích xuất sinh trắc học khuôn mặt hay danh tính cá nhân (Identity)**, tuân thủ đúng các tiêu chuẩn đạo đức AI và bảo vệ dữ liệu cá nhân."

---

### CÂU 10: Hướng phát triển tiếp theo của đề tài là gì nếu có thêm thời gian?
* **Trả lời chuẩn**:
  > "Dạ thưa Thầy/Cô, em định hướng 2 nâng cấp thực tế cao:
  > 1. **Mở rộng bộ thuộc tính**: Hiện tại hệ thống nhận diện 4 thuộc tính (Gender, Hat, Glasses, Backpack). Có thể fine-tune thêm tập nhãn PA-100K để nhận thêm tóc (Short/Long hair), áo (Long/Short sleeve), màu da giày dép...
  > 2. **Tích hợp mô hình Re-ID Embedding** để liên kết đối tượng giữa nhiều camera khác nhau trong hệ thống camera giám sát đô thị thông minh."

---

### CÂU 11: Accuracy của mô hình là bao nhiêu? Đo trên dataset nào và bằng metric gì?
* **Trả lời chuẩn**:
  > "Dạ thưa Thầy/Cô, em đánh giá mô hình trên tập validation của **PA-100K** (10,000 ảnh), sử dụng metric chuẩn của PAR là **Mean Accuracy (mA)**:
  > $$\text{mA} = \frac{1}{A}\sum_{a=1}^{A}\frac{1}{2}\left(\frac{TP_a}{P_a} + \frac{TN_a}{N_a}\right)$$
  > Kết quả cụ thể:
  >
  > | Thuộc tính | Accuracy |
  > |---|---|
  > | Giới tính (Gender) | 85.20% |
  > | Đội Mũ (Hat) | 84.70% |
  > | Đeo Kính (Glasses) | 91.00% |
  > | Đeo Balo (Backpack) | 96.70% |
  > | **Mean Accuracy (mA)** | **89.33%** |
  >
  > Mô hình được fine-tune từ ResNet50 ImageNet pretrained, huấn luyện 20 epochs trên Colab T4 GPU với 90,000 ảnh training. Chi tiết có trong script `scripts/evaluate_par.py`."

---

### CÂU 12: FPS của hệ thống là bao nhiêu? Có đạt Real-time không?
* **Trả lời chuẩn**:
  > "Dạ thưa Thầy/Cô, theo kết quả benchmark thực tế trên máy của em (i5-10300H, CPU-only):
  >
  > | Module | Latency | FPS |
  > |---|---|---|
  > | YOLOv8n Detection | 59.2 ms | 16.9 FPS |
  > | ByteTrack Tracking | 60.2 ms | 16.6 FPS |
  > | HSV Color (Trimmed Median) | 0.10 ms | 9,754 FPS |
  > | ResNet50 PAR | 43.0 ms | 23.3 FPS |
  > | **Full Pipeline** | **63.3 ms** | **~16 FPS** |
  >
  > Với cơ chế Caching (chỉ chạy ResNet mỗi 5 frame mỗi Track) và Load Budgeting (tối đa 2 CNN/frame), hệ thống duy trì mượt mà dù có 10+ người trong cùng một frame. Trên GPU GTX 1650: ước tính 20–30 FPS."

---

### CÂU 13: Tại sao nhóm dùng công thức EMA Smoothing thủ công thay vì các mô hình Deep Learning tổng hợp theo thời gian (Temporal Fusion / Spatio-Temporal Side-Tuning như trong SOTA VTFPAR++ - CVIU 2025)?
* **Trả lời chuẩn**:
  > "Dạ thưa Thầy/Cô, trong các công trình nghiên cứu SOTA gần đây về **Video-based PAR**, điển hình là **VTFPAR++** (*Nguyen et al., Computer Vision and Image Understanding - CVIU 2025* từ nhóm Event-AHU), các tác giả đã chứng minh rằng: nhận diện thuộc tính từ video đơn lẻ từng frame (Image-based PAR) thường xuyên bị nhiễu do góc quay, chuyển động nhòe (motion blur) và che khuất cục bộ. VTFPAR++ giải quyết bằng kiến trúc *Spatial-Temporal Side-Tuning* dựa trên nền tảng CLIP ViT.
  >
  > Tuy nhiên, mô hình Transformer / CLIP quá nặng nề (độ trễ >100ms/frame trên GPU phân khúc cao), không khả thi khi triển khai Edge AI trên GPU phổ thông (GTX 1650) hoặc CPU với yêu cầu xử lý đa luồng camera giám sát thời gian thực.
  > 
  > Vì vậy, cơ chế **EMA Smoothing** ($\alpha = 0.35$) mà nhóm thiết kế trong `track_memory.py` thực chất là một **dạng đơn giản hóa toán học tối ưu bậc $O(1)$** của bài toán Temporal Fusion:
  > $$P_t = \alpha \cdot \hat{P}_t + (1 - \alpha) \cdot P_{t-1}$$
  > Công thức này giúp tích lũy bằng chứng xác suất xuyên suốt trajectory của đối tượng, triệt tiêu hiện tượng nhấp nháy nhãn (label flickering) và các đột biến dương tính giả (false positive spikes) khi người quay lưng hoặc bị khuất mũ/kính tạm thời, trong khi chi phí tính toán thực tế xấp xỉ **0.00ms**."

---

### CÂU 14: Tại sao hệ thống mở rộng hỗ trợ bộ dữ liệu MSP60K (AAAI 2025 / OpenPAR) thay vì chỉ dùng PA-100K?
* **Trả lời chuẩn**:
  > "Dạ thưa Thầy/Cô, bộ dữ liệu PA-100K tuy phổ biến nhưng tồn tại điểm yếu nghiêm trọng về **mất cân bằng nhãn cực đoan (extreme class imbalance)** đối với các thuộc tính hiếm trong môi trường giám sát thực tế (như Đeo kính - Glasses tỷ lệ chỉ ~10%, Đội mũ - Hat chỉ ~15%). Khi dữ liệu ground-truth thưa thớt, F1-score của các thuộc tính này rất dễ bị sụt giảm nghiêm trọng.
  >
  > Do đó, nhóm đã tích hợp module tương thích và kịch bản chuyển đổi cho **MSP60K Benchmark Dataset** (*AAAI 2025, nhóm Event-AHU*):
  > 1. Quy mô lớn hơn: 60,122 hình ảnh đa miền với 57 thuộc tính phân bổ phong phú hơn nhiều so với 100K ảnh của PA-100K.
  > 2. Độc lập domain: Cung cấp số lượng mẫu dương tính thực tế dồi dào cho cả 4 thuộc tính mục tiêu: `female` (index 0), `hat` (index 10), `glasses` (index 11), `backpack` (index 40).
  > 3. Tái sử dụng pipeline: Module `training/dataset_msp60k.py` và script `scripts/convert_msp60k_to_pa100k.py` cho phép huấn luyện trực tiếp hoặc kết hợp cả hai bộ dữ liệu mà không làm thay đổi kiến trúc MobileNetV3/ResNet50 hiện hành."
