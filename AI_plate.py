# -*- coding: utf-8 -*-
import json
from PIL import ImageFont, ImageDraw, Image
import numpy as np
from easyocr import Reader
import cv2

# Tải tập tin JSON chứa thông tin mã tỉnh thành
with open('mabiencactinh.json', 'r', encoding='utf-8') as f:
    provinces = json.load(f)

# loading images and resizing
img = cv2.imread('./image8.jpg')
img = cv2.resize(img, (800, 600))
# load font
fontpath = "./arial.ttf"
font = ImageFont.truetype(fontpath, 32)
b, g, r, a = 0, 255, 0, 0

# bien anh thanh mau xam
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
edged = cv2.Canny(blurred, 10, 200)
cv2.waitKey(0)
cv2.destroyAllWindows()

contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

for c in contours:
    perimeter = cv2.arcLength(c, True)
    approximation = cv2.approxPolyDP(c, 0.02 * perimeter, True)
    print(approximation)
    if len(approximation) == 4:  # rectangle
        number_plate_shape = approximation
        break

(x, y, w, h) = cv2.boundingRect(number_plate_shape)
number_plate = grayscale[y:y + h, x:x + w]

# Sử dụng EasyOCR để đọc văn bản trên biển số
reader = Reader(['en'], verbose=False)
detection = reader.readtext(number_plate)

if len(detection) == 0:
    text = "Không thấy bảng số xe"
else:
    plate_text = detection[0][1]
    # Lấy mã tỉnh thành (2 chữ số đầu)
    province_code = plate_text[:2]
    province_name = provinces.get(province_code, "Không xác định")

    # Tạo văn bản hiển thị biển số và tỉnh thành
    text = f"Biển số: {plate_text}\n Tỉnh: {province_name}"
    

# Hiển thị kết quả lên hình ảnh
img_pil = Image.fromarray(img)
draw = ImageDraw.Draw(img_pil)
draw.text((200, 500), text, font=font, fill=(b, g, r, a))
img = np.array(img_pil)

# Hiển thị kết quả trên cửa sổ
cv2.imshow('Plate Detection', img)
cv2.waitKey(0)
