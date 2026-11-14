"""Hàm trích đặc trưng DÙNG CHUNG cho huấn luyện và nhận diện thời gian thực.

Quan trọng: train_model_3tx.py và predict_realtime_3tx.py đều gọi các hàm ở đây,
nên đặc trưng lúc huấn luyện và lúc chạy thật luôn giống hệt nhau.
"""
import numpy as np

TX_IDS = (1, 2, 3)
WINDOW_SEC = 2.0   # độ dài cửa sổ (giây)
STEP_SEC = 0.5     # bước trượt khi tạo mẫu huấn luyện (giây)
MIN_PKTS = 20      # số gói tối thiểu của MỖI TX trong một cửa sổ, thiếu thì bỏ cửa sổ

FEATURES_PER_TX = [
    "amp_mean",        # biên độ trung bình (qua subcarrier và thời gian)
    "amp_std",         # độ lệch chuẩn của biên độ trung bình theo thời gian
    "amp_ptp",         # khoảng dao động (max-min) của biên độ trung bình
    "sub_std_mean",    # độ lệch chuẩn theo thời gian của từng subcarrier, lấy trung bình
    "sub_std_median",  # như trên, lấy median
    "sub_std_max",     # như trên, lấy max
    "diff_mean",       # độ chênh trung bình giữa hai gói liên tiếp
    "rssi_std",        # độ lệch chuẩn RSSI
    "rssi_ptp",        # khoảng dao động RSSI
]


def feature_names():
    return [f"tx{t}_{n}" for t in TX_IDS for n in FEATURES_PER_TX]


def find_valid_columns(csi_matrix, eps=1e-6):
    """Chỉ số các cột CSI có thay đổi (bỏ cột hằng số / cột null toàn 0)."""
    return np.where(np.asarray(csi_matrix, dtype=np.float64).std(axis=0) > eps)[0]


def tx_features(amp, rssi):
    """amp: (n_gói, n_subcarrier_hợp_lệ), rssi: (n_gói,) của MỘT TX trong một cửa sổ."""
    mean_amp = amp.mean(axis=1)
    sub_std = amp.std(axis=0)
    diff_mean = np.abs(np.diff(amp, axis=0)).mean()
    return [
        mean_amp.mean(), mean_amp.std(), np.ptp(mean_amp),
        sub_std.mean(), np.median(sub_std), sub_std.max(),
        diff_mean, rssi.std(), np.ptp(rssi),
    ]


def window_features(per_tx, min_pkts=MIN_PKTS):
    """per_tx: {tx_id: (amp, rssi)}. Trả về vector đặc trưng, hoặc None nếu thiếu gói."""
    feats = []
    for t in TX_IDS:
        item = per_tx.get(t)
        if item is None or len(item[0]) < min_pkts:
            return None
        feats.extend(tx_features(*item))
    return np.array(feats, dtype=np.float64)