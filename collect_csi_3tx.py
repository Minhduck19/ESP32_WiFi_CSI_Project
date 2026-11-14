"""Thu dữ liệu CSI (3 TX - 1 RX) qua MQTT, có session_id và kiểm tra chất lượng.

Cách dùng:
  python collect_csi_3tx.py                      # hỏi từng thông tin
  python collect_csi_3tx.py --label walking --seconds 30
Mỗi lần chạy = một PHIÊN (session). Mỗi nhãn nên có nhiều phiên (>= 10 là tốt).
"""
import argparse
import csv
import json
import os
import time
from datetime import datetime

import pandas as pd
import paho.mqtt.client as mqtt

MQTT_BROKER = "127.0.0.1"
MQTT_PORT = 1883
MQTT_TOPIC = "esp32/csi_data"
EXPECTED_CSI_LEN = 128
RSSI_TOO_CLOSE = -45   # gợi ý: link mạnh hơn mức này thường bị đường truyền thẳng lấn át
RSSI_TOO_WEAK = -75    # gợi ý: link yếu hơn mức này dễ mất gói
MIN_RATE = 10.0        # gói/giây tối thiểu cho mỗi TX (gợi ý)


def next_session_id(csv_file, label):
    if not (os.path.isfile(csv_file) and os.path.getsize(csv_file) > 0):
        return f"{label}-01"
    with open(csv_file, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    if "session_id" not in header:
        raise SystemExit(
            f"[LỖI] '{csv_file}' là file cũ, không có cột session_id. "
            "Hãy dùng tên file CSV mới để không trộn hai định dạng."
        )
    ids = pd.read_csv(csv_file, usecols=["label", "session_id"])
    n = ids.loc[ids["label"] == label, "session_id"].nunique() + 1
    return f"{label}-{n:02d}"


class MultiTXCollector:
    def __init__(self, label, duration, output_file):
        self.label = label
        self.duration = duration
        self.output_file = output_file
        self.session_id = next_session_id(output_file, label)
        self.start_time = None
        self.done = False
        self.counts = {1: 0, 2: 0, 3: 0}
        self.rssi_sum = {1: 0.0, 2: 0.0, 3: 0.0}
        self.rejected = {"json": 0, "tx_id": 0, "csi_len": 0}
        self._rows_written = 0

        is_new = not (os.path.isfile(output_file) and os.path.getsize(output_file) > 0)
        self.csv_file = open(output_file, mode="a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.csv_file)
        if is_new:
            self.writer.writerow(
                ["timestamp", "session_id", "tx_id", "seq", "rssi", "label"]
                + [f"csi_{i}" for i in range(EXPECTED_CSI_LEN)]
            )

    def on_connect(self, client, userdata, flags, reason_code, properties=None):
        if reason_code == 0:
            print(f"[OK] Đã kết nối MQTT ({MQTT_BROKER}:{MQTT_PORT}). Phiên: {self.session_id}")
            client.subscribe(MQTT_TOPIC)
            self.start_time = time.time()
            print(f"[*] Đang thu {self.duration:.0f} giây cho nhãn '{self.label}'...")
        else:
            print(f"[LỖI] Kết nối MQTT thất bại: {reason_code}")

    def on_message(self, client, userdata, msg):
        if self.done or self.start_time is None:
            return
        elapsed = time.time() - self.start_time
        if elapsed >= self.duration:
            self.finish(client)
            return
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self.rejected["json"] += 1
            return
        try:
            tx_id = int(payload.get("tx_id", 0))
        except (TypeError, ValueError):
            tx_id = 0
        csi = payload.get("csi", [])
        if tx_id not in self.counts:
            self.rejected["tx_id"] += 1
            return
        if len(csi) != EXPECTED_CSI_LEN:
            self.rejected["csi_len"] += 1
            return

        rssi = payload.get("rssi", 0)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.writer.writerow([ts, self.session_id, tx_id, payload.get("seq", 0), rssi, self.label] + csi)
        self.counts[tx_id] += 1
        self.rssi_sum[tx_id] += rssi
        self._rows_written += 1
        if self._rows_written % 100 == 0:
            self.csv_file.flush()
        if self._rows_written % 100 == 0:
            print(f" -> {elapsed:5.1f}s | TX1:{self.counts[1]} TX2:{self.counts[2]} TX3:{self.counts[3]}")

    def finish(self, client):
        self.done = True
        self.csv_file.flush()
        print(f"\n[HOÀN TẤT] Phiên '{self.session_id}' đã lưu vào '{self.output_file}'.")
        self.report()
        client.disconnect()

    def report(self):
        print("--- Kiểm tra chất lượng phiên thu ---")
        for t in (1, 2, 3):
            n = self.counts[t]
            if n == 0:
                print(f" TX{t}: KHÔNG nhận được gói nào -> kiểm tra TX{t} (nguồn, kênh, MAC).")
                continue
            rate = n / self.duration
            rssi = self.rssi_sum[t] / n
            notes = []
            if rate < MIN_RATE:
                notes.append("tốc độ thấp, có thể mất gói")
            if rssi > RSSI_TOO_CLOSE:
                notes.append("quá gần RX")
            if rssi < RSSI_TOO_WEAK:
                notes.append("quá yếu")
            print(f" TX{t}: {n} gói, {rate:4.1f} gói/s, RSSI TB {rssi:6.1f} dBm"
                  + (f"  <-- {'; '.join(notes)}" if notes else ""))
        if any(self.rejected.values()):
            print(f" Gói bị loại: {self.rejected}")

    def close(self):
        self.csv_file.flush()
        self.csv_file.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label")
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--file")
    args = ap.parse_args()

    print("=== THU THẬP DỮ LIỆU WI-FI CSI (3 TX - 1 RX) ===")
    label = args.label or input("1. Nhãn (vd: empty, standing, walking): ").strip() or "unknown"
    if args.seconds is None:
        s = input("2. Thời gian thu, giây (mặc định 30): ").strip()
        duration = float(s) if s else 30.0
    else:
        duration = args.seconds
    filename = args.file or input("3. File CSV (mặc định csi_dataset_3tx_v2.csv): ").strip() or "csi_dataset_3tx_v2.csv"
    if not filename.endswith(".csv"):
        filename += ".csv"

    collector = MultiTXCollector(label, duration, filename)
    print("\nVào vị trí trong 3 giây...")
    for i in (3, 2, 1):
        print(f"  {i}...")
        time.sleep(1)

    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        client = mqtt.Client()
    client.on_connect = collector.on_connect
    client.on_message = collector.on_message
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n[DỪNG] Người dùng ngắt. Phiên này chưa đủ thời gian, nên xóa các dòng của nó.")
    finally:
        collector.close()


if __name__ == "__main__":
    main()