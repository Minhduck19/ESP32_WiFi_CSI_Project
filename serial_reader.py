import serial

# Mở cổng COM7 với baudrate 921600
ser = serial.Serial('COM7', 921600, timeout=1)

print('[*] Đang đọc dữ liệu CSI từ cổng Serial (COM7)...')
try:
  while True:
    line = ser.readline().decode('utf-8', errors='ignore').strip()
    if line:
      print(line)  # In ra dòng dữ liệu CSI thô nhận được từ board RX
except KeyboardInterrupt:
  ser.close()
  print('[*] Đã đóng kết nối Serial.')