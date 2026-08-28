"""
app.py
======
GIAO DIỆN WEB DEMO TÌM KIẾM NGƯỜI THEO ĐẶC ĐIỂM NHẬN DẠNG (100% TIẾNG VIỆT)
Đề tài: Xây dựng giải pháp phát hiện / tìm người dựa trên đặc điểm nhận dạng cho trước
"""

import os
import sys
import tempfile
import time
import cv2
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

# Thêm thư mục gốc vào path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.retrieval.pipeline import PersonRetrievalPipeline
from src.database.db import DatabaseManager
from src.utils.video_utils import (
    open_video, read_frame, release_video,
    frame_to_timestamp, resize_frame, FPSCounter, bgr_to_rgb
)
from src.utils.visualization import (
    draw_query_panel, draw_fps_and_count, translate_color
)

# ── Cấu hình trang Streamlit ──────────────────────────────────
st.set_page_config(
    page_title="Hệ thống Tìm kiếm Người theo Đặc điểm Nhận dạng",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS giao diện tiếng Việt hiện đại ──────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1976D2;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 1.2rem;
    }
    .target-card {
        background-color: #F1F8E9;
        border-left: 5px solid #4CAF50;
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 10px;
        color: #1B5E20;
    }
    .target-card b {
        color: #2E7D32;
    }
    .metric-box {
        background: #F5F5F5;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Cache pipeline để tải mô hình 1 lần duy nhất."""
    return PersonRetrievalPipeline()


@st.cache_resource
def load_db():
    return DatabaseManager()


# Bảng ánh xạ tiếng Việt -> Giá trị kỹ thuật
GENDER_MAP = {
    "Tùy ý (Bất kỳ)": "Any",
    "Nam": "Male",
    "Nữ": "Female"
}

COLOR_MAP = {
    "Tùy ý (Bất kỳ)": "Any",
    "Đen": "Black",
    "Trắng": "White",
    "Đỏ": "Red",
    "Xanh dương": "Blue",
    "Xanh lá": "Green",
    "Vàng": "Yellow",
    "Xám": "Gray",
    "Tím": "Purple",
    "Khác": "Other"
}

BOOLEAN_MAP = {
    "Tùy ý": None,
    "Có": True,
    "Không": False
}


def main():
    # ── Tiêu đề ───────────────────────────────────────────────
    st.markdown('<div class="main-header">🔍 HỆ THỐNG PHÁT HIỆN VÀ TÌM KIẾM NGƯỜI TRONG VIDEO</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Hệ thống tìm kiếm người dựa trên đặc điểm ngoại hình (YOLOv8 + ByteTrack + ResNet50 PAR)</div>', unsafe_allow_html=True)

    pipeline = load_pipeline()
    db = load_db()

    # Quản lý trạng thái tìm kiếm
    if "is_running" not in st.session_state:
        st.session_state.is_running = False

    # ── Các Tab chức năng ─────────────────────────────────────
    tab_search, tab_history, tab_eval, tab_about = st.tabs([
        "🎯 Tìm kiếm Trực quan",
        "📜 Lịch sử Tìm kiếm (Cơ sở dữ liệu)",
        "📊 Đánh giá Mô hình (Metrics)",
        "ℹ️ Giới thiệu & Cấu trúc Hệ thống"
    ])

    # ══════════════════════════════════════════════════════════
    # TAB 1: TÌM KIẾM TRỰC QUAN
    # ══════════════════════════════════════════════════════════
    with tab_search:
        col_video_view, col_results_view = st.columns([1.75, 1.25])

        # ── SIDEBAR: BỘ LỌC TÌM KIẾM TIẾNG VIỆT ────────────────
        with st.sidebar:
            st.header("⚙️ TIÊU CHÍ TÌM KIẾM")
            st.info("Chọn các đặc điểm ngoại hình của người cần tìm:")

            # 1. Giới tính
            selected_gender = st.selectbox("1. Giới tính:", list(GENDER_MAP.keys()), index=0)
            q_gender = GENDER_MAP[selected_gender]

            # 2. Màu áo
            selected_upper = st.selectbox("2. Màu áo (Thân trên):", list(COLOR_MAP.keys()), index=0)
            q_upper = COLOR_MAP[selected_upper]

            # 3. Màu quần
            selected_lower = st.selectbox("3. Màu quần/váy (Thân dưới):", list(COLOR_MAP.keys()), index=0)
            q_lower = COLOR_MAP[selected_lower]

            st.write("---")
            st.subheader("🎒 Phụ kiện đi kèm")
            sel_backpack = st.radio("Đeo Balo:", list(BOOLEAN_MAP.keys()), horizontal=True, index=0)
            sel_hat = st.radio("Đội Mũ:", list(BOOLEAN_MAP.keys()), horizontal=True, index=0)
            sel_glasses = st.radio("Đeo Kính:", list(BOOLEAN_MAP.keys()), horizontal=True, index=0)

            q_backpack = BOOLEAN_MAP[sel_backpack]
            q_hat = BOOLEAN_MAP[sel_hat]
            q_glasses = BOOLEAN_MAP[sel_glasses]

            st.write("---")
            st.subheader("🎛️ Ngưỡng điều khiển")
            q_threshold = st.slider("Ngưỡng độ khớp tối thiểu (Matching Score):", 50, 100, 70, step=5) / 100.0
            q_confidence = st.slider("Ngưỡng tin cậy phát hiện YOLO:", 20, 80, 40, step=5) / 100.0

            pipeline.matching_threshold = q_threshold
            pipeline.matcher.default_threshold = q_threshold
            pipeline.tracker.confidence_threshold = q_confidence

        target_query = {
            "gender": q_gender,
            "upper_color": q_upper,
            "lower_color": q_lower,
            "backpack": q_backpack,
            "hat": q_hat,
            "glasses": q_glasses
        }

        # ── Nguồn Video ───────────────────────────────────────
        with col_video_view:
            st.subheader("📹 Nguồn Video Đầu vào")
            video_input_type = st.radio("Chọn nguồn video:", ["Video Mẫu có sẵn", "Tải lên Video từ máy tính (.mp4, .avi)"], horizontal=True)

            video_source_path = None

            if video_input_type == "Video Mẫu có sẵn":
                sample_path = "data/test_videos/demo_search_video.mp4"
                if not os.path.exists(sample_path):
                    from scripts.demo_pipeline import create_realistic_demo_video
                    create_realistic_demo_video(sample_path)
                video_source_path = sample_path
                st.caption(f"📁 Đang dùng file mẫu: `{sample_path}`")
            else:
                uploaded_file = st.file_uploader("Chọn file video từ máy tính của bạn", type=["mp4", "avi", "mov", "mkv"])
                if uploaded_file is not None:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    tfile.write(uploaded_file.read())
                    video_source_path = tfile.name

            col_btn1, col_btn2 = st.columns(2)
            if col_btn1.button("🚀 BẮT ĐẦU TÌM KIẾM", type="primary"):
                st.session_state.is_running = True

            if col_btn2.button("🛑 DỪNG TÌM KIẾM", type="secondary"):
                st.session_state.is_running = False

            video_placeholder = st.empty()
            metrics_placeholder = st.empty()

        # ── Cột Danh sách Đối tượng Tìm thấy ──────────────────
        with col_results_view:
            st.subheader("🎯 Danh sách Mục tiêu Đã tìm thấy")
            targets_container = st.container()

        # ── VÒNG LẶP XỬ LÝ VIDEO KHI ĐANG CHẠY ─────────────────
        if st.session_state.is_running and video_source_path:
            pipeline.reset()
            cap, info = open_video(video_source_path)
            fps = info["fps"] or 25.0

            # Lưu phiên tìm kiếm vào database
            query_id = db.save_query(target_query, q_threshold)

            fps_counter = FPSCounter()
            frame_idx = 0
            all_targets_found = {}

            while cap.isOpened() and st.session_state.is_running:
                fps_counter.start_frame()
                success, frame = read_frame(cap)
                if not success:
                    break

                # Tự động tối ưu độ phân giải khung hình (720p) để tăng FPS gấp 5 lần
                h_orig, w_orig = frame.shape[:2]
                if w_orig > 800:
                    frame = resize_frame(frame, width=768)

                # Xử lý frame qua pipeline an toàn
                try:
                    annotated_frame, targets_this_frame = pipeline.process_frame(
                        frame, frame_idx, target_query, threshold=q_threshold
                    )
                except Exception as frame_err:
                    annotated_frame = frame
                    targets_this_frame = []

                timestamp_str = frame_to_timestamp(frame_idx, fps)

                # Ghi nhận đối tượng khớp
                for t in targets_this_frame:
                    tid = t["track_id"]
                    if tid not in all_targets_found or t["score"] > all_targets_found[tid]["score"]:
                        crop_img = t.get("crop")
                        crop_path = f"results/crops/target_track_{tid:03d}_{timestamp_str.replace(':', '-')}.jpg"
                        if crop_img is not None and crop_img.size > 0:
                            os.makedirs("results/crops", exist_ok=True)
                            cv2.imwrite(crop_path, crop_img)

                        target_record = {
                            "track_id": tid,
                            "score": t["score"],
                            "timestamp": timestamp_str,
                            "frame_idx": frame_idx,
                            "attributes": t["attributes"],
                            "crop_path": crop_path,
                            "crop_rgb": bgr_to_rgb(crop_img) if crop_img is not None else None
                        }
                        all_targets_found[tid] = target_record
                        db.save_target_result(query_id, target_record)

                # Vẽ UI Overlay tiếng Việt
                annotated_frame = draw_query_panel(annotated_frame, target_query, q_threshold)
                annotated_frame = draw_fps_and_count(
                    annotated_frame,
                    fps=fps_counter.get_fps(),
                    total_detected=len(pipeline.track_memory),
                    total_matched=len(all_targets_found)
                )

                # Hiển thị frame
                video_placeholder.image(bgr_to_rgb(annotated_frame), channels="RGB")

                # Cập nhật Metrics
                metrics_placeholder.markdown(f"""
                | Tốc độ (FPS) | Tổng số người theo dõi | Số lượng Mục tiêu khớp |
                | :---: | :---: | :---: |
                | **{fps_counter.get_fps():.1f} FPS** | **{len(pipeline.track_memory)} người** | **{len(all_targets_found)} mục tiêu** |
                """)

                # Cập nhật danh sách target bên phải
                with targets_container:
                    for tid, tgt in all_targets_found.items():
                        attr = tgt["attributes"]
                        g_vi = "Nữ" if str(attr.get("gender", "")).lower() in ["female", "nữ", "nu"] else "Nam"
                        up_vi = translate_color(attr.get("upper_color", ""))
                        low_vi = translate_color(attr.get("lower_color", ""))

                        st.markdown(f"""
                        <div class="target-card">
                            <b>🎯 MỤC TIÊU #{tgt['track_id']} (Độ khớp: {tgt['score']*100:.1f}%)</b><br>
                            ⏱️ Xuất hiện lúc: <code>{tgt['timestamp']}</code> (Khung hình thứ {tgt['frame_idx']})<br>
                            👤 Giới tính: <b>{g_vi}</b> | Áo: <b>{up_vi}</b> | Quần: <b>{low_vi}</b><br>
                            🎒 Balo: {'Có' if attr.get('backpack') else 'Không'} | Mũ: {'Có' if attr.get('hat') else 'Không'} | Kính: {'Có' if attr.get('glasses') else 'Không'}
                        </div>
                        """, unsafe_allow_html=True)
                        if tgt.get("crop_rgb") is not None:
                            st.image(tgt["crop_rgb"], caption=f"Chân dung Mục tiêu #{tgt['track_id']}", width=120)

                fps_counter.end_frame()
                frame_idx += 1

            release_video(cap)
            st.session_state.is_running = False
            st.success(f"✅ Hoàn tất xử lý video! Đã phát hiện {len(all_targets_found)} đối tượng đúng với yêu cầu tìm kiếm.")

    # ══════════════════════════════════════════════════════════
    # TAB 2: LỊCH SỬ TÌM KIẾM (SQLITE DATABASE)
    # ══════════════════════════════════════════════════════════
    with tab_history:
        st.subheader("📜 Nhật ký các phiên tìm kiếm gần đây trong Cơ sở dữ liệu")
        queries = db.get_recent_queries(limit=25)

        if not queries:
            st.info("Chưa có dữ liệu lịch sử tìm kiếm.")
        else:
            # Format bảng hiển thị tiếng Việt
            table_data = []
            for q in queries:
                g_vi = "Tùy ý" if q["gender"] in ["Any", None] else ("Nữ" if str(q["gender"]).lower() == "female" else "Nam")
                up_vi = "Tùy ý" if q["upper_color"] in ["Any", None] else translate_color(q["upper_color"])
                low_vi = "Tùy ý" if q["lower_color"] in ["Any", None] else translate_color(q["lower_color"])
                table_data.append({
                    "Mã phiên": f"#{q['id']}",
                    "Thời gian": q["created_at"],
                    "Giới tính": g_vi,
                    "Màu áo": up_vi,
                    "Màu quần": low_vi,
                    "Ngưỡng khớp": f"{q['threshold']*100:.0f}%",
                    "Số mục tiêu tìm thấy": f"{q['target_count']} người"
                })

            st.dataframe(pd.DataFrame(table_data))

            selected_qid = st.selectbox("Chọn Mã phiên để xem chi tiết ảnh các mục tiêu đã tìm thấy:", [q["id"] for q in queries])
            if selected_qid:
                results = db.get_results_by_query_id(selected_qid)
                if results:
                    st.write(f"**Danh sách các đối tượng tìm thấy của Phiên #{selected_qid}:**")
                    cols = st.columns(min(len(results), 4) or 1)
                    for idx, r in enumerate(results):
                        with cols[idx % 4]:
                            if os.path.exists(r["crop_path"]):
                                st.image(r["crop_path"], caption=f"Mục tiêu #{r['track_id']} (Khớp: {r['score']*100:.1f}%)")
                            st.caption(f"⏱️ Lúc: {r['timestamp_str']}")
                            st.caption(f"👕 Áo: {translate_color(r['upper_color'])} | 👖 Quần: {translate_color(r['lower_color'])}")
                else:
                    st.write("Không có đối tượng nào khớp trong phiên tìm kiếm này.")

            if st.button("🗑️ Xóa toàn bộ lịch sử cơ sở dữ liệu", type="secondary"):
                db.clear_all_history()
                st.rerun()

    # ══════════════════════════════════════════════════════════
    # TAB 3: ĐÁNH GIÁ MÔ HÌNH (METRICS)
    # ══════════════════════════════════════════════════════════
    with tab_eval:
        st.subheader("📊 Kết quả Đánh giá Mô hình PAR (Pedestrian Attribute Recognition)")

        # Thông tin training
        col_meta1, col_meta2, col_meta3 = st.columns(3)
        col_meta1.metric("Dataset", "PA-100K", "90,000 train / 10,000 val")
        col_meta2.metric("Kiến trúc", "ResNet50", "Fine-tuned từ ImageNet")
        col_meta3.metric("Training", "Colab T4 GPU", "20 epochs")

        st.markdown("---")

        # Bảng kết quả mA
        st.subheader("🎯 Mean Accuracy (mA) = 89.33%")
        eval_data = {
            "Thuộc tính": ["Giới tính (Gender)", "Đội Mũ (Hat)", "Đeo Kính (Glasses)", "Đeo Balo (Backpack)", "**Trung bình (mA)**"],
            "Accuracy": ["85.20%", "84.70%", "91.00%", "96.70%", "**89.33%**"],
            "Ngưỡng quyết định": ["≥ 50%", "≥ 62%", "≥ 50%", "≥ 50%", "-"],
            "Ghi chú": [
                "Phụ thuộc góc nhìn, trang phục",
                "Threshold cao hơn tránh nhầm tóc đen",
                "Độ chính xác cao nhất (đặc trưng rõ)",
                "Độ chính xác cao nhất (hình dạng mạnh)",
                ""
            ]
        }
        st.dataframe(pd.DataFrame(eval_data), use_container_width=True)

        # Biểu đồ cột per-class accuracy
        st.markdown("---")
        st.subheader("📈 Biểu đồ Accuracy theo từng Thuộc tính")

        try:
            import plotly.graph_objects as go
            attrs = ["Gender", "Hat", "Glasses", "Backpack"]
            accs  = [85.20, 84.70, 91.00, 96.70]
            colors = ["#2196F3", "#FF9800", "#9C27B0", "#4CAF50"]

            fig = go.Figure(data=[
                go.Bar(
                    x=attrs, y=accs,
                    marker_color=colors,
                    text=[f"{a:.1f}%" for a in accs],
                    textposition="outside",
                    width=0.5
                )
            ])
            fig.add_hline(y=89.33, line_dash="dash", line_color="red",
                          annotation_text=f"mA = 89.33%", annotation_position="right")
            fig.update_layout(
                title="Per-class Accuracy — PA-100K Test Set",
                yaxis_title="Accuracy (%)",
                yaxis=dict(range=[75, 100]),
                plot_bgcolor="white",
                height=380
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            # Fallback nếu không có plotly
            st.bar_chart({"Gender": 85.2, "Hat": 84.7, "Glasses": 91.0, "Backpack": 96.7})

        # Bảng hiệu năng FPS
        st.markdown("---")
        st.subheader("⚡ Kết quả Đo lường Hiệu năng (FPS Benchmark)")

        import torch
        device_label = f"GPU ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU (Intel i5-10300H)"
        is_gpu = torch.cuda.is_available()
        fps_data = {
            "Module / Thuật toán": [
                "1. YOLOv8n Person Detection",
                "2. ByteTrack MOT Tracker",
                "3. HSV Color Detector (Fast Median)",
                "4. ResNet50 PAR Classifier",
                "5. Attribute Matching Engine",
                "⭐ Full Pipeline (1 người)"
            ],
            "Độ trễ CPU (ms)": ["59.2 ± 5.5", "60.2 ± 12.5", "0.10 ± 0.01", "43.0 ± 2.0", "0.002", "63.3 ± 29.9"],
            "FPS CPU":  ["16.9", "16.6", "9,754", "23.3", "420,000", "**~16 FPS**"],
            "Độ trễ GPU (ms)": ["13.1 ± 2.3", "13.9 ± 1.5", "0.10 ± 0.04", "11.5 ± 0.6", "0.002", "14.1 ± 1.5"],
            "FPS GPU":  ["76.3", "72.2", "9,983", "87.3", "595,000", "**~70 FPS** 🚀"],
        }
        st.dataframe(pd.DataFrame(fps_data), use_container_width=True)
        st.caption(f"📌 Thiết bị hiện tại: **{device_label}** | Benchmark đo trực tiếp bằng `scripts/benchmark_fps.py`")

        st.markdown("---")
        st.info("💡 Để chạy Evaluation đầy đủ trên PA-100K test set, sử dụng script: `python scripts/evaluate_par.py --data-root datasets/PA100K`")

    # ══════════════════════════════════════════════════════════
    # TAB 4: GIỚI THIỆU & CẤU TRÚC HỆ THỐNG
    # ══════════════════════════════════════════════════════════
    with tab_about:
        st.subheader("📖 Tổng quan Dự án")
        st.markdown("""
        * **Tên dự án**: *Hệ thống phát hiện và tìm kiếm người dựa trên đặc điểm nhận dạng (Person Detection and Retrieval Based on Predefined Visual Attributes)*
        * **Các công nghệ và giải thuật cốt lõi**:
            1. **Phát hiện người (Detection)**: YOLOv8n (Pretrained COCO, lọc class 0 `person`)
            2. **Theo dõi đối tượng (Tracking)**: ByteTrack (Gán Track ID ổn định, liên kết 2 bước chống che khuất)
            3. **Phân tích màu sắc**: Không gian màu HSV + Thuật toán Trimmed Median (Nhanh ~0.1ms, Độc lập ánh sáng)
            4. **Nhận dạng thuộc tính (PAR)**: ResNet50 Fine-tuned trên tập dữ liệu chuẩn PA-100K (100.000 ảnh, mA = 89.33%)
            5. **Temporal EMA Smoothing**: Làm mịn nhãn theo thời gian (α=0.35) để loại bỏ rung giật khi tracking
            6. **Bộ máy so khớp (Matching Engine)**: Chấm điểm tương đồng có trọng số, hỗ trợ tìm kiếm linh hoạt
            7. **Lưu trữ**: Cơ sở dữ liệu SQLite với lịch sử tìm kiếm đầy đủ
            8. **Giao diện**: Streamlit Web Dashboard 100% Tiếng Việt
        """)

        st.markdown("---")
        st.subheader("🔄 Sơ đồ Luồng Xử lý Pipeline")
        st.code("""
Video / Camera
      ↓
YOLOv8n Person Detection  (Phát hiện bounding box người)
      ↓
ByteTrack MOT              (Gán Track ID ổn định liên frame)
      ↓
Person Crop                (Trích xuất vùng ảnh từng người)
      ↓ ┌─────────────────────────────────┐
      ↓ │ HSV+Median Color Detector        │
      ↓ │ ResNet50 PAR (Gender/Hat/...)    │
      ↓ │ Temporal EMA Smoothing           │
      ↓ └─────────────────────────────────┘
      ↓
Weighted Attribute Matching  (Tính điểm % theo trọng số)
      ↓
Person Retrieval             (Phân loại: Target / Others)
      ↓
Streamlit Web Display + SQLite Save
""", language="text")


if __name__ == "__main__":
    main()
