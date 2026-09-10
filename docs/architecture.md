# Tài liệu Kiến trúc Hệ thống
## Person Detection and Retrieval System

---

## 1. System Architecture (Kiến trúc tổng thể)

```mermaid
flowchart TD
    A[📹 Video Input\nFile / Webcam] --> B

    subgraph DET["🔍 Detection Module"]
        B[YOLOv8n\nPretrained COCO]
        B --> |class=person\nbbox, confidence| C
    end

    subgraph TRK["🎯 Tracking Module"]
        C[ByteTrack\nvia Ultralytics]
        C --> |Track ID\nổn định qua frames| D
    end

    subgraph CROP["✂️ Crop Module"]
        D[Person ROI Extractor\n224×224 resize]
    end

    D --> E
    D --> F

    subgraph PAR["🧠 PAR Module"]
        E[ResNet50\nFine-tuned PA-100K]
        E --> |Gender\nHat, Glasses\nBackpack| G
    end

    subgraph COLOR["🎨 Color Module"]
        F[HSV Analysis\nK-Means Clustering]
        F --> |Upper Color\nLower Color| G
    end

    subgraph MATCH["⚖️ Matching Engine"]
        G[Attribute Comparator]
        G --> |Score = matched/required| H{Score ≥ Threshold?}
    end

    H --> |Yes ✅| I[Highlight\nGreen Bbox]
    H --> |No ❌| J[Gray Bbox]

    I --> K
    J --> K

    subgraph OUT["📊 Output"]
        K[Streamlit GUI\nVideo + Labels]
        K --> L[(SQLite DB\nHistory)]
        K --> M[📁 Export\nCrops + CSV]
    end

    style DET fill:#1a3a5c,stroke:#4a9eff
    style TRK fill:#1a3a5c,stroke:#4a9eff
    style CROP fill:#2d1a5c,stroke:#9a4aff
    style PAR fill:#1a5c2d,stroke:#4aff9a
    style COLOR fill:#1a5c2d,stroke:#4aff9a
    style MATCH fill:#5c3a1a,stroke:#ff9a4a
    style OUT fill:#5c1a1a,stroke:#ff4a4a
```

---

## 2. Data Flow Diagram (Luồng dữ liệu)

```mermaid
flowchart LR
    V["🎬 video.mp4"] -->|cv2.VideoCapture| F["Frame\n(H×W×3 BGR)"]

    F -->|YOLOv8n\nclass=0 person| B["BBoxes\n[x1,y1,x2,y2]\nconf"]

    B -->|ByteTrack| T["Tracked Persons\n[(bbox, track_id), ...]"]

    T -->|crop_person| C["Person Crops\n(224×224×3)"]

    C -->|ResNet50 forward| PA["PAR Predictions\n{gender:0.9,\nhat:0.1,\nbackpack:0.8}"]

    C -->|HSV + K-Means| CO["Color Results\n{upper:'Black',\nlower:'Blue'}"]

    PA --> M["Merged Attributes\n{gender, upper_color,\nlower_color, hat,\nglasses, backpack}"]
    CO --> M

    M -->|Compare with Query| SC["Match Score\n3/4 = 75%"]

    SC -->|score ≥ threshold| R["✅ Matched Person\n#Track_ID"]
    SC -->|score < threshold| NR["❌ Not Matched"]
```

---

## 3. Processing Pipeline (Chi tiết từng bước)

```mermaid
sequenceDiagram
    participant U as User (Streamlit)
    participant V as Video Reader
    participant D as YOLO Detector
    participant T as ByteTrack
    participant A as Attribute Engine
    participant M as Matcher
    participant DB as SQLite

    U->>U: Chọn thuộc tính query
    U->>V: Upload video / Start webcam
    loop Mỗi frame
        V->>D: Frame (BGR numpy array)
        D->>D: Detect persons (class=0)
        D->>T: BBoxes + confidence
        T->>T: Assign/Update Track IDs
        T->>A: [(bbox, track_id), ...]
        A->>A: Crop person ROI
        A->>A: PAR Model predict
        A->>A: HSV color detect
        A->>M: Attributes dict
        M->>M: Compute match score
        M->>U: Annotated frame
        M->>DB: Save result (if matched)
    end
    U->>U: Display results
    DB->>U: Query history
```

---

## 4. PAR Model Architecture (Kiến trúc mô hình PAR)

```mermaid
flowchart TD
    I["Person Crop\n224×224×3"] --> R

    subgraph BB["ResNet50 Backbone (Frozen)"]
        R["Conv1 + BN + ReLU\n112×112×64"]
        R --> L1["Layer1: 3× Bottleneck\n56×56×256"]
        L1 --> L2["Layer2: 4× Bottleneck\n28×28×512"]
        L2 --> L3["Layer3: 6× Bottleneck\n14×14×1024"]
        L3 --> L4["Layer4: 3× Bottleneck\n7×7×2048"]
        L4 --> GAP["Global Average Pooling\n1×1×2048"]
    end

    GAP --> DROP["Dropout(0.5)"]
    DROP --> FC["FC Layer\n2048 → 512 → N_attrs"]

    subgraph HEAD["Classification Heads (Trainable)"]
        FC --> G["Gender\n2 classes\nSoftmax"]
        FC --> H["Hat\n1 class\nSigmoid"]
        FC --> GL["Glasses\n1 class\nSigmoid"]
        FC --> BP["Backpack\n1 class\nSigmoid"]
    end

    style BB fill:#1a3a1a,stroke:#4aff4a
    style HEAD fill:#3a1a3a,stroke:#ff4aff
```

**Giải thích**:
- **Backbone (ResNet50)**: Đóng băng (frozen) các layer đầu — giữ nguyên feature đã học từ ImageNet
- **Global Average Pooling**: Nén feature map 7×7×2048 → vector 2048 chiều
- **Dropout**: Tránh overfitting
- **FC + Sigmoid/Softmax**: Multi-task learning — mỗi attribute một head riêng

---

## 5. Training Pipeline (Google Colab)

```mermaid
flowchart TD
    DS["PA-100K Dataset\n100,000 ảnh + annotation.mat"] --> DL

    subgraph PREP["Data Preparation"]
        DL["DataLoader\nbatch_size=32\nshuffle=True"]
        DL --> AUG["Data Augmentation\n- RandomHorizontalFlip\n- ColorJitter\n- RandomCrop"]
        AUG --> NORM["Normalize\nmean=[0.485,0.456,0.406]\nstd=[0.229,0.224,0.225]"]
    end

    NORM --> MODEL

    subgraph TRAIN["Training Loop (30 epochs)"]
        MODEL["ResNet50 + FC Heads\n(ImageNet pretrained)"]
        MODEL --> LOSS["Multi-Task Loss\nBCE (binary attrs)\n+ CE (gender)"]
        LOSS --> OPT["Adam Optimizer\nlr=1e-4 backbone\nlr=1e-3 FC heads"]
        OPT --> SCH["LR Scheduler\nStepLR(step=10, γ=0.1)"]
        SCH --> |"next epoch"| MODEL
    end

    SCH --> EVAL

    subgraph EVAL["Evaluation"]
        EVAL["Validate on\n10,000 ảnh"]
        EVAL --> BEST{"Best mA?"}
        BEST --> |Yes| SAVE["Save par_resnet50.pth"]
        BEST --> |No| CONT["Continue training"]
    end

    SAVE --> DOWN["Download về máy\nmodels/par/par_resnet50.pth"]
```

---

## 6. Database ERD (SQLite Schema)

```mermaid
erDiagram
    DETECTIONS {
        int id PK
        text track_id
        text timestamp
        text video_source
        int frame_index
        real bbox_x1
        real bbox_y1
        real bbox_x2
        real bbox_y2
        real confidence
    }

    ATTRIBUTES {
        int id PK
        int detection_id FK
        text gender
        text upper_color
        text lower_color
        bool hat
        bool glasses
        bool backpack
    }

    SEARCH_RESULTS {
        int id PK
        int detection_id FK
        int query_id FK
        real match_score
        text image_path
        text created_at
    }

    QUERIES {
        int id PK
        text gender_query
        text upper_color_query
        text lower_color_query
        bool hat_query
        bool glasses_query
        bool backpack_query
        real threshold
        text created_at
    }

    DETECTIONS ||--o| ATTRIBUTES : "has"
    DETECTIONS ||--o{ SEARCH_RESULTS : "appears in"
    QUERIES ||--o{ SEARCH_RESULTS : "produces"
```

---

## 7. Matching Score Formula

```
                    Σ [attr_i_correct × weight_i]
  Score =  ─────────────────────────────────────────
                    Σ [attr_i_specified × weight_i]

Trong đó:
  attr_i_correct   = 1 nếu attribute thứ i dự đoán đúng với query
  attr_i_specified = 1 nếu người dùng đã chỉ định attribute thứ i
  weight_i         = trọng số của attribute (mặc định = 1.0)

Ví dụ:
  Query: {gender=Male, upper=Black, lower=Black, backpack=Yes}
  Person: {gender=Male ✓, upper=Black ✓, lower=Gray ✗, backpack=Yes ✓}

  Score = (1 + 1 + 0 + 1) / (1 + 1 + 1 + 1) = 3/4 = 75%
```

---

## 8. Temporal Smoothing & Theoretical Foundation in Video PAR (VTFPAR++ Context)

### 8.1 Thách thức trong Video-based Pedestrian Attribute Recognition (V-PAR)
Trong môi trường camera giám sát thực tế, việc dự đoán thuộc tính từng frame độc lập (Image-based PAR) thường xuyên gặp các hiện tượng:
- **Nhấp nháy nhãn (Label Flickering):** Người đi bộ xoay người, đổi góc nhìn hoặc bị che khuất một phần (occlusion) khiến thuộc tính như mũ, kính, balo biến mất trong vài frame rồi xuất hiện lại.
- **Dương tính giả đột biến (False Positive Spikes):** Đốm sáng, bóng đổ hoặc chuyển động mờ (motion blur) gây ra nhiễu dự đoán cục bộ.

### 8.2 Cơ sở học thuật SOTA: VTFPAR++ (CVIU 2025)
Công trình SOTA mới nhất về Video PAR từ nhóm Event-AHU — **VTFPAR++** (*Nguyen et al., "Video-based Pedestrian Attribute Recognition via Spatial-Temporal Side Tuning", Computer Vision and Image Understanding - CVIU 2025*):
- Đề xuất kiến trúc **Spatial-Temporal Side-Tuning** dựa trên CLIP ViT để học tương quan thời gian giữa chuỗi các frames video.
- **Hạn chế đối với Edge Deployment:** Mô hình Transformer/ViT này yêu cầu tài nguyên tính toán rất lớn (độ trễ >100ms/frame trên GPU cao cấp, bộ nhớ VRAM lớn), hoàn toàn không khả thi cho bài toán giám sát đa luồng thời gian thực trên phần cứng phổ thông (GTX 1650 hoặc CPU).

### 8.3 Giải pháp tối ưu biên của PDR-System: $O(1)$ EMA Temporal Fusion
Hệ thống PDR giải quyết bài toán trên bằng cách trừu tượng hóa quá trình Temporal Fusion thành cơ chế **Exponential Moving Average (EMA) Smoothing** trong `modules/track_memory.py`:
$$P_t = \alpha \cdot \hat{P}_t + (1 - \alpha) \cdot P_{t-1}$$
- **Trọng số cập nhật:** $\alpha = 0.35$, cân bằng giữa khả năng thích ứng với góc nhìn mới ($35\%$) và độ ổn định lịch sử ($65\%$).
- **Chi phí tính toán:** Đạt độ phức tạp $O(1)$ với độ trễ xấp xỉ **0.00ms**, không tiêu tốn GPU.
- **Cơ chế Caching & Load Budgeting:** Giảm tải tới $80\%$ số lần gọi mạng CNN nặng mà vẫn duy trì độ chính xác và độ mượt mà của nhãn theo thời gian.

### 8.4 Mở rộng sang Dataset MSP60K (AAAI 2025)
Để khắc phục hiện tượng mất cân bằng mẫu hiếm (như F1 của `glasses` và `hat` trong PA-100K), hệ thống hỗ trợ tích hợp bộ dữ liệu **MSP60K** (AAAI 2025, 60,122 ảnh, 57 thuộc tính) thông qua:
- `training/dataset_msp60k.py`: DataLoader tối ưu hỗ trợ trích xuất 4 thuộc tính cốt lõi và Data Augmentation (Perspective, Erasing, ColorJitter).
- `scripts/convert_msp60k_to_pa100k.py`: Công cụ chuyển đổi định dạng `dataset_ms_split1.pkl` sang `annotation.mat` và `dataset_converted_4attrs.pkl`.
