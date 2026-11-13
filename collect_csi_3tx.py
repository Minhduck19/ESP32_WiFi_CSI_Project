import csv
import json
import os
import time
from datetime import datetime
import paho.mqtt.client as mqtt

MQTT_BROKER = "127.0.0.1"
MQTT_PORT = 1883
MQTT_TOPIC = "esp32/csi_data"
EXPECTED_CSI_LEN = 128


class MultiTXCollector:
    def __init__(self, label: str, target_total: int, output_file: str):
        self.label = label
        self.target_total = target_total
        self.output_file = output_file
        self.total_count = 0
        self.tx_counts = {1: 0, 2: 0, 3: 0}
        self.start_time = None

        file_exists = os.path.isfile(self.output_file) and os.path.getsize(self.output_file) > 0
        self.csv_file = open(self.output_file, mode="a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.csv_file)

        if not file_exists:
            header = ["timestamp", "tx_id", "seq", "rssi", "label"] + [
                f"csi_{i}" for i in range(EXPECTED_CSI_LEN)
            ]
            self.writer.writerow(header)

    def on_connect(self, client, userdata, flags, reason_code, properties=None):
        if reason_code == 0:
            print(f"[OK] Đã kết nối MQTT Broker ({MQTT_BROKER}:{MQTT_PORT})")
            client.subscribe(MQTT_TOPIC)
            print(f"[*] Bắt đầu thu thập {self.target_total} gói tổng cho nhãn '{self.label}'...")
            self.start_time = time.time()
        else:
            print(f"[LỖI] Kết nối MQTT thất bại, mã lỗi: {reason_code}")

    def on_message(self, client, userdata, msg):
        if self.total_count >= self.target_total:
            return

        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            tx_id = int(payload.get("tx_id", 0))
            csi_array = payload.get("csi", [])

            if tx_id not in (1, 2, 3) or len(csi_array) != EXPECTED_CSI_LEN:
                return

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            seq = payload.get("seq", 0)
            rssi = payload.get("rssi", 0)

            row = [timestamp, tx_id, seq, rssi, self.label] + csi_array
            self.writer.writerow(row)

            self.total_count += 1
            self.tx_counts[tx_id] += 1

            if self.total_count % 30 == 0 or self.total_count == self.target_total:
                elapsed = max(time.time() - self.start_time, 0.1) if self.start_time else 0.1
                rate = self.total_count / elapsed
                print(
                    f" -> Tổng: {self.total_count}/{self.target_total} "
                    f"(TX1:{self.tx_counts[1]} | TX2:{self.tx_counts[2]} | TX3:{self.tx_counts[3]}) "
                    f"| Tốc độ: {rate:.1f} gói/s"
                )

            if self.total_count >= self.target_total:
                print(f"\n[HOÀN TẤT] Đã lưu dữ liệu nhãn '{self.label}' vào '{self.output_file}'.")
                client.disconnect()

        except Exception:
            pass

    def close(self):
        self.csv_file.flush()
        self.csv_file.close()


def main():
    print("=== THU THẬP DỮ LIỆU WI-FI CSI (3 TX - 1 RX) ===")
    label = input("1. Nhập tên nhãn hành động (vd: empty, standing, walking): ").strip() or "unknown"
    samples_str = input("2. Nhập tổng số gói muốn thu (mặc định 1500 ~ 25 giây): ").strip()
    target_total = int(samples_str) if samples_str.isdigit() else 1500
    filename = input("3. Tên file CSV lưu trữ (mặc định csi_dataset_3tx.csv): ").strip() or "csi_dataset_3tx.csv"
    if not filename.endswith(".csv"):
        filename += ".csv"

    print("\nChuẩn bị vào vị trí đo trong 3 giây...")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    collector = MultiTXCollector(label=label, target_total=target_total, output_file=filename)

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
        print("\n[DỪNG] Người dùng ngắt thu thập.")
    finally:
        collector.close()


if __name__ == "__main__":
    main()