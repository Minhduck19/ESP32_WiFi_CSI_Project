import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC

# ==========================================
# 1. TẠO DỮ LIỆU GIẢ LẬP (MOCK CSI DATA)
# ==========================================


def generate_mock_csi_data(num_samples_per_session=500):
  """Giả lập dữ liệu CSI thu từ 3 TX và 1 RX với 3 nhãn: empty, static, walking"""
  np.random.seed(42)
  labels = ['empty', 'static', 'walking']
  sessions = [f'session_{i}' for i in range(1, 10)]  # 9 phiên đo khác nhau

  data = []
  num_subcarriers = 52  # Số subcarrier mang dữ liệu hữu ích chuẩn 802.11n HT20

  for session in sessions:
    label = np.random.choice(labels)
    start_time = 1730000000000

    for tx_id in [1, 2, 3]:
      mac_address = f'24:0A:C4:00:0{tx_id}:00'

      for i in range(num_samples_per_session):
        ts = start_time + i * 20  # Mỗi gói cách nhau 20ms (~50 gói/giây)

        # Mô phỏng biên độ CSI dao động theo hoạt động
        if label == 'empty':
          base_amp = np.random.normal(20, 1, num_subcarriers)
        elif label == 'static':
          base_amp = np.random.normal(25, 3, num_subcarriers) + np.sin(
              i * 0.1
          )
        else:  # walking
          base_amp = (
              np.random.normal(30, 8, num_subcarriers)
              + np.sin(i * 0.5) * 5
              + np.random.normal(0, 2, num_subcarriers)
          )

        row = {
            'timestamp_ms': ts,
            'tx_id': tx_id,
            'mac': mac_address,
            'rssi': int(np.random.normal(-50, 5)),
            'channel': 6,
            'label': label,
            'session_id': session,
        }
        # Thêm biên độ các subcarrier vào dòng dữ liệu
        for sc in range(num_subcarriers):
          row[f'amp_{sc}'] = max(1.0, base_amp[sc])

        data.append(row)

  df = pd.DataFrame(data)
  print(f'[*] Đã tạo dataset giả lập thành công với {len(df)} dòng dữ liệu!')
  return df


# ==========================================
# 2. TIỀN XỬ LÝ DỮ LIỆU (Hampel Filter lọc nhiễu gai)
# ==========================================


def hampel_filter(series, window_size=5, n_sigmas=3):
  """Loại bỏ các điểm giá trị gãy/gai đột ngột (outlier)"""
  rolling_median = series.rolling(window=window_size, center=True).median()
  difference = (series - rolling_median).abs()
  median_deviation = difference.rolling(window=window_size, center=True).median()
  threshold = n_sigmas * median_deviation
  outlier_idx = difference > threshold
  filtered_series = series.copy()
  filtered_series[outlier_idx] = rolling_median[outlier_idx]
  return filtered_series.fillna(series)


def preprocess_csi_data(df):
  """Thực hiện lọc nhiễu theo từng đường truyền phát (tx_id)"""
  print('[*] Đang thực hiện tiền xử lý và lọc nhiễu tín hiệu...')
  amp_cols = [col for col in df.columns if col.startswith('amp_')]

  processed_dfs = []
  for tx_id, group in df.groupby('tx_id'):
    group = group.copy()
    for col in amp_cols:
      group[col] = hampel_filter(group[col], window_size=5)
    processed_dfs.append(group)

  return pd.concat(processed_dfs).sort_values('timestamp_ms').reset_index(drop=True)


# ==========================================
# 3. TRÍCH XUẤT ĐẶC TRƯNG VỚI CỬA SỔ TRƯỢT
# ==========================================


def extract_features_sliding_window(
    df, window_size_ms=2000, step_size_ms=1000
):
  """Cắt cửa sổ trượt (2 giây, bước trượt 1 giây) và tính toán đặc trưng thống kê"""
  print('[*] Đang trích xuất đặc trưng thống kê qua cửa sổ trượt...')
  amp_cols = [col for col in df.columns if col.startswith('amp_')]
  feature_rows = []

  for session_id, session_group in df.groupby('session_id'):
    label = session_group['label'].iloc[0]
    min_ts = session_group['timestamp_ms'].min()
    max_ts = session_group['timestamp_ms'].max()

    curr_start = min_ts
    while curr_start + window_size_ms <= max_ts:
      curr_end = curr_start + window_size_ms

      window_data = session_group[
          (session_group['timestamp_ms'] >= curr_start)
          & (session_group['timestamp_ms'] < curr_end)
      ]

      if len(window_data) > 5:
        feature_dict = {
            'session_id': session_id,
            'label': label,
            'start_time': curr_start,
        }

        # Trích xuất đặc trưng cho từng TX
        for tx_id in [1, 2, 3]:
          tx_data = window_data[window_data['tx_id'] == tx_id]
          if len(tx_data) > 0:
            tx_amps = tx_data[amp_cols]

            mean_vals = tx_amps.mean().values
            std_vals = tx_amps.std().values
            var_vals = tx_amps.var().values
            min_vals = tx_amps.min().values
            max_vals = tx_amps.max().values

            for i in range(len(amp_cols)):
              feature_dict[f'tx{tx_id}_mean_{i}'] = mean_vals[i]
              feature_dict[f'tx{tx_id}_std_{i}'] = std_vals[i]
              feature_dict[f'tx{tx_id}_var_{i}'] = var_vals[i]
              feature_dict[f'tx{tx_id}_max_{i}'] = max_vals[i]
              feature_dict[f'tx{tx_id}_min_{i}'] = min_vals[i]

        feature_rows.append(feature_dict)

      curr_start += step_size_ms

  feature_df = pd.DataFrame(feature_rows)
  print(
      f'[*] Trích xuất đặc trưng hoàn tất! Tổng số mẫu cửa sổ:'
      f' {len(feature_df)}'
  )
  return feature_df


# ==========================================
# 4. HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH MACHINE LEARNING
# ==========================================


def train_and_evaluate_models(feature_df):
  """Huấn luyện và đánh giá mô hình Random Forest & SVM phân loại hành động"""
  print(
      '[*] Đang phân tách tập dữ liệu huấn luyện / kiểm tra theo phiên đo và'
      ' chạy ML...'
  )

  feature_cols = [
      col
      for col in feature_df.columns
      if col not in ['session_id', 'label', 'start_time']
  ]
  X = feature_df[feature_cols].fillna(0)
  y = feature_df['label']

  # Chia train/test theo session để đảm bảo tính khách quan khoa học
  unique_sessions = feature_df['session_id'].unique()
  train_sessions = unique_sessions[:6]
  test_sessions = unique_sessions[6:]

  train_idx = feature_df['session_id'].isin(train_sessions)
  test_idx = feature_df['session_id'].isin(test_sessions)

  X_train, y_train = X[train_idx], y[train_idx]
  X_test, y_test = X[test_idx], y[test_idx]

  # 1. Random Forest (Mô hình chính)
  rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
  rf_model.fit(X_train, y_train)
  rf_pred = rf_model.predict(X_test)

  print('\n--- KẾT QUẢ ĐÁNH GIÁ RANDOM FOREST ---')
  print(f'Accuracy (Độ chính xác): {accuracy_score(y_test, rf_pred) * 100:.2f}%')
  print('Classification Report:\n', classification_report(y_test, rf_pred))
  print('Confusion Matrix:\n', confusion_matrix(y_test, rf_pred))

  # 2. Support Vector Machine (SVM - Mô hình đối chứng)
  svm_model = SVC(kernel='rbf', C=1.0, random_state=42)
  svm_model.fit(X_train, y_train)
  svm_pred = svm_model.predict(X_test)

  print('\n--- KẾT QUẢ ĐÁNH GIÁ SVM ---')
  print(
      f'Accuracy (Độ chính xác): {accuracy_score(y_test, svm_pred) * 100:.2f}%'
  )
  print('Classification Report:\n', classification_report(y_test, svm_pred))
  print('Confusion Matrix:\n', confusion_matrix(y_test, svm_pred))


# ==========================================
# CHẠY TOÀN BỘ PIPELINE THỰC NGHIỆM
# ==========================================
if __name__ == '__main__':
  print('=== BẮT ĐẦU CHẠY PIPELINE MOCK CSI (NGÀY 5 & 6) ===\n')

  # Bước 1: Tạo dữ liệu giả lập
  raw_df = generate_mock_csi_data()

  # Bước 2: Tiền xử lý tín hiệu
  clean_df = preprocess_csi_data(raw_df)

  # Bước 3: Cửa sổ trượt & Trích xuất đặc trưng
  features_df = extract_features_sliding_window(clean_df)

  # Bước 4: Huấn luyện & Đánh giá mô hình
  train_and_evaluate_models(features_df)

  print('\n=== HOÀN TẤT THỰC THI THÀNH CÔNG! ===')