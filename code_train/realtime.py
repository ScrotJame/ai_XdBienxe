import cv2
import pytesseract
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from PIL import Image

# Đường dẫn đến Tesseract OCR nếu cần thiết
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Tải mô hình đã huấn luyện
def load_keras_model(model_name):
    with open(f'./{model_name}.json', 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights(f"./{model_name}.weights.h5")
    return model

# Chuẩn bị mô hình
model = load_keras_model('model_LicensePlate')

# Chức năng xử lý khung hình và nhận diện biển số
def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Giả lập phân đoạn ký tự
    char_images = segment_characters(img_binary)
    license_plate_text = ''
    
    for char_img in char_images:
        img_resized = cv2.resize(char_img, (28, 28))
        img_resized = np.expand_dims(img_resized, axis=-1)
        img_resized = np.expand_dims(img_resized, axis=0)
        prediction = model.predict(img_resized)
        predicted_char = np.argmax(prediction, axis=1)
        license_plate_text += chr(predicted_char[0])

    return license_plate_text

# Phân đoạn các ký tự từ biển số
def segment_characters(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_images = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 15:  # Điều chỉnh kích thước lọc phù hợp
            char_images.append(img[y:y+h, x:x+w])
    return sorted(char_images, key=lambda x: x.shape[1], reverse=True)

# Kết nối tới camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Nhận diện biển số
    plate_number = process_frame(frame)
    
    # Hiển thị kết quả nhận diện trên khung hình
    cv2.putText(frame, f"License Plate: {plate_number}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Hiển thị khung hình
    cv2.imshow('License Plate Detection', frame)

    # Thoát nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
