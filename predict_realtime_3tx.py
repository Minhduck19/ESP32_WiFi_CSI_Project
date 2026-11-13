import json
from collections import deque
import joblib
import numpy as np
import paho.mqtt.client as mqtt

MQTT_BROKER = "127.0.0.1"
MQTT_PORT = 1883
MQTT_TOPIC = "esp32/csi_data"
MODEL_FILE = "csi_model_3tx.pkl"


def hampel_filter_1d(series: np.ndarray, window_size: int = 3, n_sigmas: float = 3.0) -> np.ndarray:
    filtered = series.copy()
    k = 1.4826
    for i in range(window_size, len(series) - window_size):
        window = series[i - window_size : i + window_size + 1]
        median = np.median(window)
        mad = k * np.median(np.abs(window - median))
        if mad > 0 and np.abs(series[i] - median) > n_sigmas * mad:
            filtered[i] = median
    return filtered


def extract_tx_feature(buffer_deque: deque) -> np.ndarray:
    mat = np.array(buffer_deque, dtype=np.float32)
    for col in range(mat.shape[1]):
        mat[:, col] = hampel_filter_1d(mat[:, col])
    return np.concatenate([np.mean(mat, axis=0), np.std(mat, axis=0), np.ptp(mat, axis=0)])


class RealtimeCSIPredictor:
    def __init__(self, model_path: str):
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.scaler = artifact["scaler"]
        self.win_size = artifact["window_per_tx"]
        self.buffers = {
            1: deque(maxlen=self.win_size),
            2: deque(maxlen=self.win_size),
            3: deque(maxlen=self.win_size),
        }
        self.packet_counter = 0

    def on_connect(self, client, userdata, flags, reason_code, properties=None):
        if reason_code == 0:
            print(f"[OK] Sẵn sàng nhận diện thời gian thực từ '{MQTT_TOPIC}'...\n")
            client.subscribe(MQTT_TOPIC)

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            tx_id = int(payload.get("tx_id", 0))
            csi = payload.get("csi", [])

            if tx_id not in (1, 2, 3) or len(csi) != 128:
                return

            self.buffers[tx_id].append(csi)
            self.packet_counter += 1

            # Dự đoán mỗi 15 gói khi cả 3 hàng đợi đã đầy
            if self.packet_counter % 15 == 0 and all(
                len(self.buffers[i]) == self.win_size for i in (1, 2, 3)
            ):
                f1 = extract_tx_feature(self.buffers[1])
                f2 = extract_tx_feature(self.buffers[2])
                f3 = extract_tx_feature(self.buffers[3])

                X_live = np.concatenate([f1, f2, f3]).reshape(1, -1)
                X_scaled = self.scaler.transform(X_live)

                pred_label = self.model.predict(X_scaled)[0]
                probs = self.model.predict_proba(X_scaled)[0]
                confidence = np.max(probs) * 100

                print(f"[DỰ ĐOÁN] Hành động: {pred_label.upper():<12} | Độ tin cậy: {confidence:.1f}%")

        except Exception:
            pass


def main():
    predictor = RealtimeCSIPredictor(MODEL_FILE)
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        client = mqtt.Client()

    client.on_connect = predictor.on_connect
    client.on_message = predictor.on_message
    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    client.loop_forever()


if __name__ == "__main__":
    main()