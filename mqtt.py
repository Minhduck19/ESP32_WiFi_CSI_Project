import json
import random
import time
import paho.mqtt.client as mqtt

# Khởi tạo client MQTT (tương thích cả paho-mqtt v1 và v2)
client = mqtt.Client()

# Kết nối trực tiếp qua IP 127.0.0.1 thay vì chữ 'localhost'
try:
  client.connect("127.0.0.1", 1883, 60)
  client.loop_start()  # CỰC KỲ QUAN TRỌNG: Chạy vòng lặp nền để gửi tin nhắn
  print("[*] Kết nối MQTT Broker thành công!")
except Exception as e:
  print(f"[!] Không thể kết nối tới Mosquitto Broker: {e}")

labels = ["empty", "static", "walking"]

print("[*] Đang bắn dữ liệu MQTT giả lập lên dashboard...")
try:
  while True:
    payload = {
        "label": random.choice(labels),
        "confidence": round(random.uniform(0.75, 0.98), 2),
        "ts": int(time.time() * 1000),
    }
    # Publish tin nhắn lên topic
    client.publish("ips/har/state", json.dumps(payload))
    print(f"Đã gửi: {payload}")
    time.sleep(1)
except KeyboardInterrupt:
  client.loop_stop()
  client.disconnect()
  print("[*] Đã dừng script.")