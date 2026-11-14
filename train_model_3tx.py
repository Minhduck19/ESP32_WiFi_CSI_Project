"""Huấn luyện nhận diện hoạt động từ CSI 3 TX, ĐÁNH GIÁ THEO PHIÊN THU CHƯA THẤY.

  python train_model_3tx.py --csv csi_dataset_3tx.csv
  python train_model_3tx.py --csv csi_dataset_3tx.csv --map standing=still,empty=still,walking=motion

Đánh giá: phiên thứ k của MỖI nhãn được giữ lại làm tập kiểm tra, huấn luyện trên các
phiên còn lại (leave-one-session-round-out). Không bao giờ chia ngẫu nhiên các cửa sổ.
File CSV cũ (không có session_id) vẫn dùng được: phiên được tách theo khoảng trống > 5 giây.
"""
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import csi_features as F


def load(csv_path, gap_sec=5.0):
    df = pd.read_csv(csv_path).copy()
    csi_cols = [c for c in df.columns if c.startswith("csi_")]
    ts = pd.to_datetime(df["timestamp"])
    df["t"] = (ts - ts.min()).dt.total_seconds()
    df = df.sort_values("t").reset_index(drop=True)
    if "session_id" not in df.columns:
        k = df.groupby("label")["t"].transform(lambda s: (s.diff().fillna(0) > gap_sec).cumsum())
        df["session_id"] = df["label"] + "#" + k.astype(str)
        print("[i] File không có session_id: tách phiên theo khoảng trống thời gian.")
    starts = df.groupby(["label", "session_id"])["t"].min().reset_index()
    starts["fold"] = starts.groupby("label")["t"].rank(method="first").astype(int) - 1
    df = df.merge(starts[["label", "session_id", "fold"]], on=["label", "session_id"])
    return df, csi_cols


def build_windows(df, valid_cols, label_map, window, step, min_pkts):
    X, y, folds = [], [], []
    skipped = 0
    for (label, sid, fold), g in df.groupby(["label", "session_id", "fold"]):
        per = {}
        for t in F.TX_IDS:
            gt = g[g["tx_id"] == t]
            per[t] = (gt["t"].to_numpy(), gt[valid_cols].to_numpy(np.float32), gt["rssi"].to_numpy(float))
        s, t_end = g["t"].min(), g["t"].max()
        while s + window <= t_end + 1e-9:
            win = {}
            for t, (ts, amp, rs) in per.items():
                i0, i1 = np.searchsorted(ts, s, "left"), np.searchsorted(ts, s + window, "left")
                win[t] = (amp[i0:i1], rs[i0:i1])
            feats = F.window_features(win, min_pkts)
            if feats is None:
                skipped += 1
            else:
                X.append(feats)
                y.append(label_map.get(label, label))
                folds.append(fold)
            s += step
    return np.array(X), np.array(y), np.array(folds), skipped


def new_model():
    return RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0, n_jobs=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="csi_dataset_3tx_v2.csv")
    ap.add_argument("--out", default="csi_model_3tx.pkl")
    ap.add_argument("--window", type=float, default=F.WINDOW_SEC)
    ap.add_argument("--step", type=float, default=F.STEP_SEC)
    ap.add_argument("--min-pkts", type=int, default=F.MIN_PKTS)
    ap.add_argument("--map", default="", help="gộp nhãn, vd: standing=still,empty=still,walking=motion")
    args = ap.parse_args()
    label_map = dict(p.split("=") for p in args.map.split(",") if p)

    print(f"[1/4] Đọc '{args.csv}'...")
    df, csi_cols = load(args.csv)
    sess = df.groupby("label")["session_id"].nunique()
    print(" Số phiên mỗi nhãn:", {str(k): int(v) for k, v in sess.items()}, "| gói mỗi TX:", df["tx_id"].value_counts().sort_index().to_dict())

    valid_idx = F.find_valid_columns(df[csi_cols].to_numpy())
    valid_cols = [csi_cols[i] for i in valid_idx]
    print(f" Giữ {len(valid_cols)}/{len(csi_cols)} cột CSI (bỏ {len(csi_cols) - len(valid_cols)} cột hằng/null).")

    print("[2/4] Tạo cửa sổ theo thời gian và trích đặc trưng...")
    X, y, folds, skipped = build_windows(df, valid_cols, label_map, args.window, args.step, args.min_pkts)
    if len(X) == 0:
        raise SystemExit("[LỖI] Không tạo được cửa sổ nào: dữ liệu quá ngắn hoặc có TX không gửi gói. "
                         "Kiểm tra lại từng TX bằng báo cáo cuối mỗi phiên thu.")
    classes = sorted(map(str, set(y)))
    print(f" {len(X)} cửa sổ ({X.shape[1]} đặc trưng), bỏ {skipped} cửa sổ thiếu gói. "
          f"Phân bố: { {c: int((y == c).sum()) for c in classes} }")

    print("[3/4] Đánh giá trên phiên chưa thấy...")
    uniq = sorted(set(folds))
    if len(uniq) < 2:
        print(" [CẢNH BÁO] Chỉ có 1 phiên mỗi nhãn, không thể đánh giá trung thực. Hãy thu thêm phiên.")
    else:
        all_true, all_pred = [], []
        for f in uniq:
            tr, te = folds != f, folds == f
            if set(y[tr]) != set(classes):
                print(f" [bỏ qua fold {f}: tập huấn luyện thiếu lớp]")
                continue
            m = new_model().fit(X[tr], y[tr])
            p = m.predict(X[te])
            all_true += list(y[te]); all_pred += list(p)
            print(f"  giữ lại phiên số {f + 1} của mỗi nhãn: accuracy {accuracy_score(y[te], p) * 100:.1f}%")
        print(f"\n=== ACCURACY TRÊN PHIÊN CHƯA THẤY: {accuracy_score(all_true, all_pred) * 100:.1f}% ===")
        maj = pd.Series(all_true).value_counts(normalize=True).max() * 100
        print(f"(đoán lớp phổ biến nhất sẽ được {maj:.1f}%)\n")
        print(classification_report(all_true, all_pred, digits=3))
        print("Confusion matrix (hàng = thật, cột = đoán), thứ tự:", classes)
        print(confusion_matrix(all_true, all_pred, labels=classes))
        if len(uniq) < 4:
            print(f"\n[Lưu ý] Mới có {len(uniq)} phiên mỗi nhãn nên con số trên chỉ là ước lượng thô.")

    print("\n[4/4] Huấn luyện mô hình cuối trên toàn bộ dữ liệu và lưu...")
    model = new_model().fit(X, y)
    joblib.dump({
        "model": model, "classes": [str(c) for c in model.classes_], "valid_cols": [int(i) for i in valid_idx],
        "csi_len": len(csi_cols), "window_sec": args.window, "min_pkts": args.min_pkts,
        "feature_names": F.feature_names(), "label_map": label_map,
    }, args.out)
    print(f" Đã lưu '{args.out}'.")


if __name__ == "__main__":
    main()