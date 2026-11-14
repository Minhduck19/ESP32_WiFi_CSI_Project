"""Nhận diện hoạt động thời gian thực từ CSI 3 TX, publish kết quả lên MQTT cho dashboard.

  python predict_realtime_3tx.py --model csi_model_3tx.pkl
Đọc 'esp32/csi_data', publish {"label","confidence","ts"} lên 'ips/har/state'.
"""
import argparse
import json
import time
from collections import deque

import joblib
import numpy as np

import csi_features as F


class Predictor:
    """Không phụ thuộc MQTT, nên có thể kiểm thử bằng cách phát lại dữ liệu."""

    def __init__(self, artifact, smooth=3):
        self.model = artifact["model"]
        self.classes = list(artifact["classes"])
        self.valid_cols = np.array(artifact["valid_cols"])
        self.csi_len = artifact["csi_len"]
        self.window = artifact["window_sec"]
        self.min_pkts = artifact["min_pkts"]
        self.buf = {t: deque() for t in F.TX_IDS}
        self.history = deque(maxlen=smooth)

    def ingest(self, tx_id, csi, rssi, now):
        if tx_id not in self.buf or len(csi) != self.csi_len:
            return False
        self.buf[tx_id].append((now, np.asarray(csi, dtype=np.float32)[self.valid_cols], float(rssi)))
        for d in self.buf.values():                      # bỏ gói cũ hơn cửa sổ, kể cả TX đã im
            while d and d[0][0] < now - self.window:
                d.popleft()
        return True

    def counts(self):
        return {t: len(d) for t, d in self.buf.items()}

    def predict(self):
        per = {}
        for t, d in self.buf.items():
            if len(d) < self.min_pkts:
                return None
            per[t] = (np.stack([x[1] for x in d]), np.array([x[2] for x in d]))
        feats = F.window_features(per, self.min_pkts)
        if feats is None:
            return None
        self.history.append(self.model.predict_proba(feats.reshape(1, -1))[0])
        avg = np.mean(self.history, axis=0)               # làm mượt qua vài lần dự đoán gần nhất
        i = int(np.argmax(avg))
        return self.classes[i], float(avg[i])


def main():
    import paho.mqtt.client as mqtt

    ap = argparse.ArgumentParser()
    ap.add_argument("--broker", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--model", default="csi_model_3tx.pkl")
    ap.add_argument("--topic-in", default="esp32/csi_data")
    ap.add_argument("--topic-out", default="ips/har/state")
    ap.add_argument("--interval", type=float, default=0.5, help="chu kỳ dự đoán (giây)")
    ap.add_argument("--smooth", type=int, default=3)
    args = ap.parse_args()

    pred = Predictor(joblib.load(args.model), smooth=args.smooth)
    state = {"last": 0.0, "last_status": 0.0, "bad": 0}

    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        client = mqtt.Client()

    def on_connect(c, userdata, flags, reason_code, properties=None):
        if reason_code == 0:
            print(f"[OK] Đang nhận '{args.topic_in}', publish lên '{args.topic_out}'. Lớp: {pred.classes}")
            c.subscribe(args.topic_in)
        else:
            print(f"[LỖI] MQTT: {reason_code}")

    def on_message(c, userdata, msg):
        now = time.time()
        try:
            p = json.loads(msg.payload.decode("utf-8"))
            ok = pred.ingest(int(p.get("tx_id", 0)), p.get("csi", []), p.get("rssi", 0), now)
        except (ValueError, TypeError, UnicodeDecodeError):
            ok = False
        if not ok:
            state["bad"] += 1
            return
        if now - state["last"] < args.interval:
            return
        state["last"] = now
        res = pred.predict()
        if res is None:
            if now - state["last_status"] > 2:
                state["last_status"] = now
                print(f"[chờ đủ gói] số gói trong cửa sổ: {pred.counts()} (cần >= {pred.min_pkts} mỗi TX)")
            return
        label, conf = res
        c.publish(args.topic_out, json.dumps({"label": label, "confidence": round(conf, 3), "ts": int(now * 1000)}))
        print(f"[DỰ ĐOÁN] {label.upper():<10} | tin cậy {conf * 100:5.1f}% | gói/TX {list(pred.counts().values())}")

    client.on_connect, client.on_message = on_connect, on_message
    client.connect(args.broker, args.port, keepalive=60)
    client.loop_forever()


if __name__ == "__main__":
    main()