import cv2
import easyocr
import time
import json
import sys
import threading
import numpy as np

# Đọc dữ liệu từ file mabiencactinh.json
with open('mabiencactinh.json', 'r', encoding='utf-8') as f:
    province_data = json.load(f)

# Đảm bảo hệ thống hỗ trợ UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Khởi tạo EasyOCR Reader một lần duy nhất
reader = easyocr.Reader(['vi'], gpu=True, verbose=False)  # Sử dụng GPU nếu có

# Hàm xử lý OCR trong luồng riêng
def ocr_thread_function():
    global frame_to_process, ocr_results, lock, processing

    while True:
        lock.acquire()
        if frame_to_process is not None and not processing:
            processing = True
            frame = frame_to_process.copy()
            frame_to_process = None
            lock.release()

            # Xử lý OCR
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Áp dụng bộ lọc để giảm nhiễu
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            # Tăng cường độ tương phản bằng CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

            # Sử dụng EasyOCR để nhận diện văn bản
            results = reader.readtext(gray, detail=1, paragraph=False)

            lock.acquire()
            ocr_results = results
            processing = False
            lock.release()
        else:
            lock.release()
            time.sleep(0.01)  # Ngủ ngắn để giảm tải CPU

# Hàm phát hiện biển số xe (sử dụng Haar Cascade)
def detect_license_plate(frame, plate_cascade):
    # Chuyển đổi sang grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Áp dụng bộ lọc để giảm nhiễu
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Phát hiện biển số
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 20))
    return plates

# Khởi tạo biến toàn cục cho luồng OCR
frame_to_process = None
ocr_results = []
processing = False
lock = threading.Lock()

# Tải Haar Cascade cho biển số xe
plate_cascade_path = cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
plate_cascade = cv2.CascadeClassifier(plate_cascade_path)

# Kiểm tra tải Cascade thành công
if plate_cascade.empty():
    print("Không thể tải Haar Cascade. Vui lòng kiểm tra đường dẫn.")
    sys.exit()

# Khởi động luồng OCR
ocr_thread = threading.Thread(target=ocr_thread_function, daemon=True)
ocr_thread.start()

# Mở video và thiết lập độ phân giải
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Tăng độ phân giải
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # Tăng độ phân giải

# Thiết lập xử lý mỗi N khung hình (để tối ưu tốc độ)
PROCESS_EVERY_N_FRAMES = 3
frame_count = 0

# Danh sách các plate đã được nhận diện và thời gian hiển thị
recognized_plates = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Phát hiện biển số xe trong khung hình
    plates = detect_license_plate(frame, plate_cascade)

    current_time = time.time()

    for (x, y, w, h) in plates:
        # Vẽ khung biển số
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Cắt vùng biển số để xử lý
        plate_img = frame[y:y+h, x:x+w]

        # Cải thiện chất lượng ảnh biển số
        plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        plate_gray = cv2.GaussianBlur(plate_gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        plate_gray = clahe.apply(plate_gray)
        _, plate_binary = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Gán khung hình để xử lý OCR nếu không đang xử lý
        lock.acquire()
        if not processing:
            frame_to_process = plate_img.copy()
        lock.release()

    # Lấy kết quả OCR
    lock.acquire()
    results = ocr_results.copy()
    ocr_results = []
    lock.release()

    for (bbox, text, prob) in results:
        if prob > 0.5:
            # Vẽ khung bao quanh văn bản
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))
            cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)
            cv2.putText(frame, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Lấy mã tỉnh từ biển số
            plate_prefix = ''.join(filter(str.isdigit, text[:2]))  # Lấy 2 chữ số đầu tiên

            # Kiểm tra nếu mã tỉnh hợp lệ
            province = province_data.get(plate_prefix, "Khong xac dinh")

            if province != "Khong xac dinh":
                info_text = f"Biển số: {text} - Tỉnh: {province}"
            else:
                info_text = f"Biển số: {text} - Tỉnh: Khong xac dinh"

            # Thêm vào danh sách plate đã nhận diện
            recognized_plates.append((info_text, current_time))

    # Hiển thị thông tin biển số đã nhận diện trong vòng 3 giây
    for plate_info, timestamp in recognized_plates.copy():
        if current_time - timestamp <= 3:
            # Hiển thị thông tin trên khung hình
            # Đặt vị trí hiển thị thông tin khác nhau nếu có nhiều biển số
            index = recognized_plates.index((plate_info, timestamp))
            cv2.putText(frame, plate_info, (10, 50 + 30 * index), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            # Xóa plate_info sau 3 giây
            recognized_plates.remove((plate_info, timestamp))

    # Hiển thị video với biển số nhận diện
    cv2.imshow('License Plate Recognition', frame)

    # Kiểm tra nếu phím 'q' được nhấn để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
