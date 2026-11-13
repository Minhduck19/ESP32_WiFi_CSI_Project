import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CSV_FILE = "csi_dataset_3tx.csv"
MODEL_OUTPUT = "csi_model_3tx.pkl"
WINDOW_PER_TX = 12  # 12 gói mỗi TX (~0.6 giây ở tần số 20Hz)
STEP_PER_TX = 5     # Bước trượt 5 gói


def hampel_filter_1d(series: np.ndarray, window_size: int = 4, n_sigmas: float = 3.0) -> np.ndarray:
    filtered = series.copy()
    k = 1.4826
    for i in range(window_size, len(series) - window_size):
        window = series[i - window_size : i + window_size + 1]
        median = np.median(window)
        mad = k * np.median(np.abs(window - median))
        if mad > 0 and np.abs(series[i] - median) > n_sigmas * mad:
            filtered[i] = median
    return filtered


def clean_tx_matrix(raw_csi: np.ndarray) -> np.ndarray:
    cleaned = np.zeros_like(raw_csi)
    for col in range(raw_csi.shape[1]):
        cleaned[:, col] = hampel_filter_1d(raw_csi[:, col])
    return cleaned


def compute_stats(win: np.ndarray) -> np.ndarray:
    return np.concatenate([np.mean(win, axis=0), np.std(win, axis=0), np.ptp(win, axis=0)])


def build_3tx_dataset(df: pd.DataFrame):
    csi_cols = [f"csi_{i}" for i in range(128)]
    X_all, y_all = [], []

    for label, group in df.groupby("label", sort=False):
        tx_data = {}
        for tx_id in (1, 2, 3):
            sub = group[group["tx_id"] == tx_id][csi_cols].to_numpy(dtype=np.float32)
            tx_data[tx_id] = clean_tx_matrix(sub)

        min_len = min(len(tx_data[1]), len(tx_data[2]), len(tx_data[3]))
        if min_len < WINDOW_PER_TX:
            print(f"[CẢNH BÁO] Nhãn '{label}' không đủ dữ liệu cho cả 3 TX (min={min_len}), bỏ qua.")
            continue

        for start in range(0, min_len - WINDOW_PER_TX + 1, STEP_PER_TX):
            f1 = compute_stats(tx_data[1][start : start + WINDOW_PER_TX])
            f2 = compute_stats(tx_data[2][start : start + WINDOW_PER_TX])
            f3 = compute_stats(tx_data[3][start : start + WINDOW_PER_TX])

            X_all.append(np.concatenate([f1, f2, f3]))
            y_all.append(label)

    return np.array(X_all), np.array(y_all)


def main():
    print(f"[1/4] Đang đọc dữ liệu từ '{CSV_FILE}'...")
    df = pd.read_csv(CSV_FILE)
    print(f" -> Tổng gói tin: {len(df)} | Phân bố nhãn:\n{df['label'].value_counts()}\n")

    print("[2/4] Đang lọc nhiễu Hampel và ghép đặc trưng từ 3 TX...")
    X, y = build_3tx_dataset(df)
    print(f" -> Tạo được {len(X)} mẫu tổng hợp (mỗi mẫu {X.shape[1]} đặc trưng).\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("[3/4] Đang huấn luyện mô hình SVM (RBF Kernel)...")
    model = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    print(f"\n=== ĐỘ CHÍNH XÁC MÔ HÌNH 3 TX: {accuracy_score(y_test, y_pred) * 100:.2f}% ===\n")
    print(classification_report(y_test, y_pred))
    print("Ma trận nhầm lẫn (Confusion Matrix):")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "window_per_tx": WINDOW_PER_TX,
            "classes": list(model.classes_),
        },
        MODEL_OUTPUT,
    )
    print(f"\n[4/4] Đã lưu mô hình hoàn chỉnh tại '{MODEL_OUTPUT}'.")


if __name__ == "__main__":
    main()