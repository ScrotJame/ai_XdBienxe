import cv2
import easyocr
import time
import json
import sys

# Đọc dữ liệu từ file mabiencactinh.json
with open('mabiencactinh.json', 'r', encoding='utf-8') as f:
    province_data = json.load(f)

sys.stdout.reconfigure(encoding='utf-8')
reader = easyocr.Reader(['vi'], verbose=False)  

# Mở video
cap = cv2.VideoCapture(0)
frame_rate = 15
frame_time = 5 / frame_rate

while True:
    start_time = time.time()  # Lưu thời gian bắt đầu của vòng lặp
    
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray)

    # Vẽ khung bao quanh biển số xe và hiển thị kết quả
    for (bbox, text, prob) in results:
        if prob > 0.5: 
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(frame, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            plate_prefix = text[:2]  # Lấy 2 chữ số đầu tiên (mã tỉnh)
            
            # Kiểm tra nếu mã tỉnh hợp lệ
            province = province_data.get(plate_prefix, "Khong xac dinh")
            if province != "Khong xac dinh":
                # Nếu mã tỉnh hợp lệ, hiển thị tên tỉnh
                cv2.putText(frame, f"Tinh: {province}", (top_left[0], top_left[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                # Nếu không phải là mã tỉnh hợp lệ
                cv2.putText(frame, "Khong xac dinh", (top_left[0], top_left[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Hiển thị video với biển số nhận diện
    cv2.imshow('License Plate Recognition', frame)

    # Kiểm tra nếu phím 'q' được nhấn để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Tính toán thời gian để điều chỉnh tốc độ FPS
    elapsed_time = time.time() - start_time
    delay_time = max(0, frame_time - elapsed_time)  # Đảm bảo không có độ trễ âm
    time.sleep(delay_time)  # Điều chỉnh độ trễ giữa các khung hình

cap.release()
cv2.destroyAllWindows()
