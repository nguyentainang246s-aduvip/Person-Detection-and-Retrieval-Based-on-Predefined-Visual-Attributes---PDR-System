"""
scripts/convert_msp60k_to_pa100k.py
===================================
Công cụ chuyển đổi tập dữ liệu MSP60K (AAAI 2025 / OpenPAR) sang định dạng tương thích
với pipeline huấn luyện PA-100K (training/dataset_pa100k.py và training/train_par.py).

CÁCH DÙNG:
    python scripts/convert_msp60k_to_pa100k.py --input_pkl path/to/dataset_ms_split1.pkl --output_dir datasets/MSP60K_Converted/
"""

import os
import sys
import pickle
import argparse
import numpy as np

# Reconfigure stdout for utf-8 if needed on Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Đảm bảo scipy được nạp nếu có
try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Ánh xạ từ MSP60K sang PA-100K 26 thuộc tính chuẩn
# PA-100K 26 attributes:
# 0: Female, 7: Backpack, 10: Hat, 11: Glasses
MSP_TO_PA100K_MAP = {
    0: 0,   # Female -> Female
    10: 10, # Hat -> Hat
    11: 11, # Glasses -> Glasses
    40: 7,  # Backpack -> Backpack
}


def convert_msp60k_to_mat(input_pkl: str, output_dir: str):
    """
    Đọc file dataset_ms_split1.pkl và xuất ra annotation.mat tương thích với dataset_pa100k.py.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_mat_path = os.path.join(output_dir, "annotation.mat")
    out_pkl_path = os.path.join(output_dir, "dataset_converted_4attrs.pkl")

    print(f"[*] Bắt đầu đọc dữ liệu MSP60K từ: {input_pkl}")
    with open(input_pkl, "rb") as f:
        data = pickle.load(f)

    image_names = data.image_name
    labels = data.label  # shape (N, 57)
    partition = data.partition

    print(f"[+] Tổng số ảnh trong MSP60K: {len(image_names)}")
    print(f"[+] Phân chia dữ liệu: Train={len(partition.get('train', []))}, Val={len(partition.get('val', []))}, Test={len(partition.get('test', []))}")

    # Chuẩn bị dữ liệu cho 3 split
    splits_data = {}
    stats = {}

    for s_name in ["train", "val", "test"]:
        indices = partition.get(s_name, [])
        names = [image_names[i] for i in indices]
        sub_labels_57 = labels[indices]

        # Tạo ma trận nhãn 26 cột cho PA-100K
        pa_labels = np.zeros((len(indices), 26), dtype=np.uint8)
        for msp_idx, pa_idx in MSP_TO_PA100K_MAP.items():
            pa_labels[:, pa_idx] = sub_labels_57[:, msp_idx]

        splits_data[s_name] = {
            "images_name": names,
            "label_26": pa_labels,
            "label_4": pa_labels[:, [0, 10, 11, 7]]  # female, hat, glasses, backpack
        }

        # Thống kê phân bố lớp
        stats[s_name] = {
            "total": len(indices),
            "female": int(pa_labels[:, 0].sum()),
            "hat": int(pa_labels[:, 10].sum()),
            "glasses": int(pa_labels[:, 11].sum()),
            "backpack": int(pa_labels[:, 7].sum())
        }

    # Xuất file .mat (chuẩn PA-100K)
    if HAS_SCIPY:
        mat_dict = {
            "train_images_name": np.array(splits_data["train"]["images_name"], dtype=object),
            "val_images_name": np.array(splits_data["val"]["images_name"], dtype=object),
            "test_images_name": np.array(splits_data["test"]["images_name"], dtype=object),
            "train_label": splits_data["train"]["label_26"],
            "val_label": splits_data["val"]["label_26"],
            "test_label": splits_data["test"]["label_26"],
            "attributes": np.array([
                "Female", "AgeOver60", "Age18-60", "AgeLess18", "Front", "Back", "Side",
                "Backpack", "HandBag", "ShoulderBag", "Hat", "Glasses", "HoldObjectsInFront",
                "ShortSleeve", "LowerStripe", "LowerPattern", "LongCoat", "Trousers", "Shorts",
                "LowerPattern", "UpperPattern", "Shorts", "Skirt&Dress", "boots", "LongHair", "BlackHair"
            ], dtype=object)
        }
        sio.savemat(out_mat_path, mat_dict)
        print(f"[+] Đã xuất file annotation chuẩn MATLAB: {out_mat_path}")
    else:
        print("[!] Không tìm thấy thư viện scipy, bỏ qua xuất file .mat.")

    # Xuất file .pkl nhẹ (tối ưu nạp nhanh cho Python)
    with open(out_pkl_path, "wb") as f:
        pickle.dump({
            "splits": splits_data,
            "target_attributes": ["female", "hat", "glasses", "backpack"],
            "stats": stats
        }, f)
    print(f"[+] Đã xuất file metadata tối ưu Python: {out_pkl_path}")

    # In báo cáo phân phối mẫu
    print("\n" + "=" * 70)
    print("📊 THỐNG KÊ MẪU DỮ LIỆU ĐÃ CHUYỂN ĐỔI (MSP60K -> PDR TARGETS)")
    print("=" * 70)
    print(f"{'Tập':<8} | {'Tổng mẫu':<10} | {'Nữ (Female)':<12} | {'Mũ (Hat)':<10} | {'Kính (Glasses)':<15} | {'Balo (Backpack)'}")
    print("-" * 70)
    for s_name, s in stats.items():
        print(f"{s_name:<8} | {s['total']:<10} | {s['female']:<12} | {s['hat']:<10} | {s['glasses']:<15} | {s['backpack']}")
    print("-" * 70 + "\n")


def create_sample_msp60k_pkl(save_path: str = "data/sample_dataset_ms_split1.pkl", num_samples: int = 500):
    """Tạo file pickle mẫu mô phỏng MSP60K để test script chuyển đổi."""
    from types import SimpleNamespace
    np.random.seed(42)

    data = SimpleNamespace()
    data.image_name = [f"sample_msp_{i:05d}.jpg" for i in range(num_samples)]
    data.attributes = [f"attr_{i}" for i in range(57)]
    data.attributes[0] = "Female"
    data.attributes[10] = "Hat"
    data.attributes[11] = "Glasses"
    data.attributes[40] = "Backpack"

    data.label = np.random.binomial(1, 0.25, (num_samples, 57))
    # Tăng tỷ lệ mẫu positive cho 4 thuộc tính cốt lõi
    data.label[:, 0] = np.random.binomial(1, 0.48, num_samples)
    data.label[:, 10] = np.random.binomial(1, 0.22, num_samples)
    data.label[:, 11] = np.random.binomial(1, 0.31, num_samples)
    data.label[:, 40] = np.random.binomial(1, 0.27, num_samples)

    n_train = int(num_samples * 0.7)
    n_val = int(num_samples * 0.15)
    indices = list(range(num_samples))
    np.random.shuffle(indices)

    data.partition = {
        "train": indices[:n_train],
        "val": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:]
    }

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(data, f)

    print(f"[+] Đã tạo file MSP60K mẫu thành công tại: {save_path} ({num_samples} mẫu)")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuyển đổi dataset MSP60K sang định dạng PA-100K")
    parser.add_argument("--input_pkl", type=str, default="data/sample_dataset_ms_split1.pkl", help="Đường dẫn file pkl MSP60K")
    parser.add_argument("--output_dir", type=str, default="datasets/MSP60K_Converted", help="Thư mục lưu kết quả")
    parser.add_argument("--create_sample", action="store_true", help="Tạo file pkl mẫu để kiểm thử")
    args = parser.parse_args()

    if args.create_sample or not os.path.exists(args.input_pkl):
        create_sample_msp60k_pkl(args.input_pkl)

    convert_msp60k_to_mat(args.input_pkl, args.output_dir)
