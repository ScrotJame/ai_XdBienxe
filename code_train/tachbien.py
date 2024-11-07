import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import pytesseract   
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"           
from PIL import Image
plt.style.use('dark_background')
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('mabiencactinh.json', 'r', encoding='utf-8') as f:
    provinces = json.load(f)

rasm = cv2.imread('bienso/image8.jpg')

height, width, channel = rasm.shape
#xac dinh anh
plt.figure(figsize=(12, 10))
plt.imshow(rasm, cmap='gray')
plt.axis('on')
plt.savefig('biensotrain/Car.png',bbox_inches = 'tight')
#plt.show()
print(height)
print(width)
print(channel)
# bien anh thanh den tranggray = cv2.cvtColor(rasm, cv2.COLOR_BGR2GRAY)
gray = cv2.cvtColor(rasm, cv2.COLOR_BGR2GRAY)

# Định nghĩa phần tử cấu trúc
structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Áp dụng phép toán hình thái top-hat và black-hat
imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)

# Kết hợp với ảnh xám ban đầu
imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

# Áp dụng cân bằng histogram để tăng độ tương phản
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
gray_clahe = clahe.apply(gray)

# Hiển thị và lưu kết quả
plt.figure(figsize=(12, 10))
plt.imshow(gray, cmap='gray')
plt.axis('off')
plt.savefig('biensotrain/Car_Xam.png', bbox_inches='tight')
#plt.show()
#anh den trang de nhan biet bien va xe 
img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)  #loc Gauss

img_thresh = cv2.adaptiveThreshold(
    img_blurred, 
    maxValue=255.0, 
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY_INV, 
    blockSize=19, 
    C=9
)
img_thresh=cv2.equalizeHist(img_thresh)
plt.figure(figsize=(12, 10))
plt.imshow(img_thresh, cmap='gray')
plt.axis('off')
plt.savefig('biensotrain/Car_Locvien.png',bbox_inches = 'tight')
#plt.show()
#doi sang dang duong vien 
contours, _= cv2.findContours(
    img_thresh, 
    mode=cv2.RETR_LIST, 
    method=cv2.CHAIN_APPROX_SIMPLE
)

temp_result = np.zeros((height, width, channel), dtype=np.uint8)

cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))
plt.figure(figsize=(12, 10))
plt.imshow(temp_result)
plt.axis('off')
plt.savefig('biensotrain/Car_Vientrang.png',bbox_inches = 'tight')
#plt.show()

temp_result = np.zeros((height, width, channel), dtype=np.uint8)

contours_dict = []

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=2)
    
    # chen lenh
    contours_dict.append({
        'contour': contour,
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'cx': x + (w / 2),
        'cy': y + (h / 2)
    })

plt.figure(figsize=(12, 10))
plt.imshow(temp_result, cmap='gray')
plt.axis('off')
plt.savefig('biensotrain/Car_Lochop.png',bbox_inches = 'tight')
#plt.show()

#tach lay cac o nho hon 
MIN_AREA = 50
MIN_WIDTH, MIN_HEIGHT = 2, 8
MIN_RATIO, MAX_RATIO = 0.25, 1.0

possible_contours = []

cnt = 0
for d in contours_dict:
    area = d['w'] * d['h']
    ratio = d['w'] / d['h']
    
    if area > MIN_AREA \
    and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
    and MIN_RATIO < ratio < MAX_RATIO:
        d['idx'] = cnt
        cnt += 1
        possible_contours.append(d)

temp_result = np.zeros((height, width, channel), dtype=np.uint8)

for d in possible_contours:
    cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
    cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)

plt.figure(figsize=(12, 10))
plt.imshow(temp_result, cmap='gray')
plt.axis('off')
plt.savefig('biensotrain/Car_Lochop2.png',bbox_inches = 'tight')
#plt.show()

#lay o bien so xe
MAX_DIAG_MULTIPLYER = 5 # 5
MAX_ANGLE_DIFF = 11.0 # 12.0
MAX_AREA_DIFF = 0.2 # 0.5
MAX_WIDTH_DIFF = 0.8 #0.8
MAX_HEIGHT_DIFF = 0.2 #0.2
MIN_N_MATCHED = 3 # 3

def find_chars(contour_list):
    matched_result_idx = []
    
    for d1 in contour_list:
        matched_contours_idx = []
        for d2 in contour_list:
            if d1['idx'] == d2['idx']:
                continue

            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])

            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
            if dx == 0:
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(dy / dx))
            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
            width_diff = abs(d1['w'] - d2['w']) / d1['w']
            height_diff = abs(d1['h'] - d2['h']) / d1['h']

            if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
            and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
            and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx'])

        # noi vien 
        matched_contours_idx.append(d1['idx'])

        if len(matched_contours_idx) < MIN_N_MATCHED:
            continue

        matched_result_idx.append(matched_contours_idx)

        unmatched_contour_idx = []
        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx:
                unmatched_contour_idx.append(d4['idx'])

        unmatched_contour = np.take(possible_contours, unmatched_contour_idx)
        
        # de quy o day
        recursive_contour_list = find_chars(unmatched_contour)
        
        for idx in recursive_contour_list:
            matched_result_idx.append(idx)

        break

    return matched_result_idx
    
result_idx = find_chars(possible_contours)

matched_result = []
for idx_list in result_idx:
    matched_result.append(np.take(possible_contours, idx_list))

# hinh dung duong vien
temp_result = np.zeros((height, width, channel), dtype=np.uint8)

for r in matched_result:
    for d in r:
        cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)

plt.figure(figsize=(12, 10))
plt.imshow(temp_result, cmap='gray')
plt.axis('off')
plt.savefig('biensotrain/Car_Locvungbien.png',bbox_inches = 'tight')
plt.show()

#ap dung duogn vien cho anh goc 
result_idx = find_chars(possible_contours)

matched_result = []
for idx_list in result_idx:
    matched_result.append(np.take(possible_contours, idx_list))
temp_result = np.zeros((height, width, channel), dtype=np.uint8)

for r in matched_result:
    for d in r:
        cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
        cv2.rectangle(rasm, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(0, 0, 255), thickness=2)

plt.figure(figsize=(12, 10))
plt.imshow(rasm, cmap='gray')
plt.axis('off')
plt.savefig('biensotrain/Car_Locgoc.png',bbox_inches = 'tight')
plt.show()

#cat so ra khoi hinh
PLATE_WIDTH_PADDING = 1.3  # 1.3
PLATE_HEIGHT_PADDING = 1.5  # 1.5
MIN_PLATE_RATIO = 3
MAX_PLATE_RATIO = 10

plate_imgs = []
plate_infos = []

for i, matched_chars in enumerate(matched_result):
    sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

    plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
    plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2
    
    plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING
    
    sum_height = 0
    for d in sorted_chars:
        sum_height += d['h']

    plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)
    
    triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
    triangle_hypotenus = np.linalg.norm(
        np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) - 
        np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
    )
    
    angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))
    
    rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)
    
    img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height))
    
    img_cropped = cv2.getRectSubPix(
        img_rotated, 
        patchSize=(int(plate_width), int(plate_height)), 
        center=(int(plate_cx), int(plate_cy))
    )
    
    if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[0] > MAX_PLATE_RATIO:
        continue
    
    # Convert to grayscale (if not already) and apply histogram equalization
    if len(img_cropped.shape) == 3:  # Check if the image is colored
        img_cropped_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
    else:
        img_cropped_gray = img_cropped

    # Apply histogram equalization
    img_cropped_eq = cv2.equalizeHist(img_cropped_gray)
    #img_cropped_eq = cv2.calcHist(img_cropped_gray)
    plate_imgs.append(img_cropped_eq)
    plate_infos.append({
        'x': int(plate_cx - plate_width / 2),
        'y': int(plate_cy - plate_height / 2),
        'w': int(plate_width),
        'h': int(plate_height)
    })
    
    plt.subplot(len(matched_result), 1, i+1)
    plt.imshow(img_cropped_eq, cmap='gray')  # Use the equalized image for display
    plt.axis('off')
    plt.savefig('biensotrain/Car_daonguocbien.png', bbox_inches='tight')
    plt.show()

    longest_idx, longest_text = -1, 0
    plate_chars = []

for i, plate_img in enumerate(plate_imgs):
    plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)

    # Tăng độ tương phản bằng histogram
    plate_img = cv2.equalizeHist(plate_img)

    _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Tìm đường viền lại lần nữa 
    contours, _ = cv2.findContours(plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    
    plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
    plate_max_x, plate_max_y = 0, 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        area = w * h
        ratio = w / h
        # Lọc các đường viền dựa trên kích thước điển hình của số biển số xe
        if area > MIN_AREA and w > MIN_WIDTH and h > MIN_HEIGHT and MIN_RATIO < ratio < MAX_RATIO:
            if x < plate_min_x:
                plate_min_x = x
            if y < plate_min_y:
                plate_min_y = y
            if x + w > plate_max_x:
                plate_max_x = x + w
            if y + h > plate_max_y:
                plate_max_y = y + h 
                
    img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]
    
    img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)
    _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    plt.subplot(len(plate_imgs), 1, i+1)
    plt.imshow(img_result, cmap='gray')
    plt.axis('off')
    plt.savefig('biensotrain/Car_Sotrang.png', bbox_inches='tight')
    plt.show()

img = 255 - img_result
plt.imshow(img, 'gray')
plt.axis('off')
plt.savefig('biensotrain/Car-Soden.png', bbox_inches='tight')
plt.show()
def find_contours(dimensions, img) :

    # Tim tat ca vien trong anh
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Truy xuat thu tiem nang la so
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    
    # kiem tra lan luot 5-15 duong vien lon nhat cho bien hoac so xe
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    
    ii = cv2.imread('biensotrain/sobien.jpg')
    
    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs :
        # phat hien duong vien cac ky tu
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        
        # kiem tra kich thuoc duong vien de loc cac ky tu 
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
            x_cntr_list.append(intX) #luu tru toa do cua duong vien ky tu 

            char_copy = np.zeros((44,24))
            # trich tung ky tu
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (20, 40))
            
            cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)
            plt.imshow(ii, cmap='gray')
            plt.axis('off')

            # Dao nguoc mau
            char = cv2.subtract(255, char)

            # thay doi kich thuoc anh thanh 24x44 
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy) # danh sach luu tru hinh anh nhi phan cua ky tu
            
    # Tra ve cac ky tu tu ben trai 
            
    plt.show()
    # luu tru ky tu da cat
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])# luu tru hinh anh
    img_res = np.array(img_res_copy)

    return img_res

# tim ky tu trong hinh da cat
def segment_characters(image) :

    # xu ly hinh anh da cat
    img_lp = cv2.resize(image, (333, 75))
    img_lp = cv2.equalizeHist(img_lp)

    LP_WIDTH = img_lp.shape[0]
    LP_HEIGHT = img_lp.shape[1]

    # tao vien trang
    img_lp[0:3,:] = 255
    img_lp[:,0:3] = 255
    img_lp[72:75,:] = 255
    img_lp[:,330:333] = 255

    # Uoc tinh kich thuoc
    dimensions = [LP_WIDTH/6,
                       LP_WIDTH/2,
                       LP_HEIGHT/10,
                       2*LP_HEIGHT/3]
    plt.imshow(img_lp, cmap='gray')
    plt.axis('off')
    plt.show()
    cv2.imwrite('biensotrain/sobien.jpg', img_lp)
    

    # nhan duong vien trong bien so xe da cat
    char_list = find_contours(dimensions, img_lp)

    return char_list
char = segment_characters(img)
plt.style.use('ggplot')
for i in range(len(char)):
    plt.subplot(1, len(char), i+1)
    plt.imshow(char[i], cmap='gray')
    plt.axis('off')
#tach cac so 
plt.savefig('biensotrain/Car_biencuoi.png',bbox_inches = 'tight')

import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
from keras.models import Sequential, model_from_json  # Thêm dòng này để import model_from_json
import tensorflow as tf 

"""
def f1score(y, y_pred):
    return f1_score(y, tf.math.argmax(y_pred, axis=1), average='micro') 

def custom_f1score(y, y_pred):
    return tf.py_function(f1score, (y, y_pred), tf.double)
        """

# Đào tạo và Xác thực Xử lý dữ liệu
train_datagen = ImageDataGenerator(rescale=1./255, width_shift_range=0.1, height_shift_range=0.1)
path = './biensotrain/data'
train_generator = train_datagen.flow_from_directory(
    path + '/train',  
    target_size=(28, 28),
    batch_size=1,
    class_mode='sparse'
)

validation_generator = train_datagen.flow_from_directory(
    path + '/val',  
    target_size=(28, 28),
    batch_size=1,
    class_mode='sparse'
)

# Đăng ký lớp F1Score để lưu và tải
@tf.keras.utils.register_keras_serializable()
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', dtype=tf.float32):
        super(F1Score, self).__init__(name=name, dtype=dtype)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)  # Chuyển kiểu dữ liệu cho y_true
        y_pred = tf.cast(y_pred, tf.int32)  # Chuyển kiểu dữ liệu cho y_pred

        tp = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred) & tf.equal(y_true, 1), tf.float32))
        fp = tf.reduce_sum(tf.cast(tf.equal(y_true, 0) & tf.equal(y_pred, 1), tf.float32))
        fn = tf.reduce_sum(tf.cast(tf.equal(y_true, 1) & tf.equal(y_pred, 0), tf.float32))

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-7)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-7)
        return 2 * (precision * recall) / (precision + recall + 1e-7)

    def reset_state(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)

# Model lưu trữ
def store_keras_model(model, model_name):
    model_json = model.to_json()  # tuần tự hóa mô hình thành JSON
    with open("./{}.json".format(model_name), "w") as json_file:
        json_file.write(model_json)
    model.save_weights("./{}.weights.h5".format(model_name))  # Lưu trọng số với tên đúng
    print("Saved model to disk")

def load_keras_model(model_name):
    json_file = open('./{}.json'.format(model_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("./{}.weights.h5".format(model_name))
    return model

# Mô hình đào tạo
K.clear_session()
model = Sequential()
model.add(Conv2D(16, (22,22), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (16,16), activation='relu', padding='same'))
model.add(Conv2D(64, (8,8), activation='relu', padding='same'))
model.add(Conv2D(128, (4,4), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(4,4)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu'))
model.add(Dense(36, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0001), metrics=[F1Score()])

model.summary()

# Callback để dừng sớm
class StopTrainingCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_f1_score') and logs['val_f1_score'] > 0.99:
            self.model.stop_training = True

# Train
batch_size = 1
callbacks = [StopTrainingCallback()]
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    epochs=80,
    verbose=1,
    callbacks=callbacks
)




#Lưu mô hình
store_keras_model(model, 'model_LicensePlate')

#Tải mô hình được đào tạo trước
pre_trained_model = load_keras_model('model_LicensePlate')
model = pre_trained_model 

#Kiểm tra xem mô hình
print(pre_trained_model.summary())

#Dự đoán đầu ra và trình bày kết quả
def fix_dimension(img):
    new_img = np.zeros((28,28,3))
    for i in range(3):
        new_img[:,:,i] = img 
    return new_img

def show_results(char):  # Thêm tham số char vào
    dic = {}
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i, c in enumerate(characters):
        dic[i] = c

    output = []
    for i, ch in enumerate(char):  # Lặp lại các ký tự
        img_ = cv2.resize(ch, (28, 28), interpolation=cv2.INTER_AREA)
        img = fix_dimension(img_)
        img = img.reshape(1, 28, 28, 3)  # Chuẩn bị hình ảnh cho mô hình
        predict_x = model.predict(img)[0]
        y_ = np.argmax(predict_x, axis=0)
        character = dic[y_]
        output.append(character)  # Lưu trữ kết quả trong một danh sách
    
    plate_number = ''.join(output)
    return plate_number

print(show_results(char))

#Các ký tự phân đoạn và giá trị dự đoán của chúng
plt.figure(figsize=(10,6))
for i,ch in enumerate(char):
    img = cv2.resize(ch, (28,28), interpolation=cv2.INTER_AREA)
    plt.subplot(3,4,i+1)
    plt.imshow(img,cmap='gray')
    title_obj = plt.title(f'predicted: {show_results(char)[i]}')
    plt.setp(title_obj, color='black')
    plt.axis('off')
plt.show()

#Xem kết quả thông qua Pytesseract
img_1 = Image.fromarray(img_result)
txt = pytesseract.image_to_string(img_1)
print("Biển xe : ", txt)

province_code = txt[:2]

# Kiểm tra và in ra tên tỉnh tương ứng

if province_code in provinces:
    print("Tỉnh: ", provinces[province_code])
else:
    print("Tỉnh không xác định")

char = segment_characters(img_result)

txt = ''
for i in range(len(char)):
    plt.subplot(1, len(char), i + 1)
    plt.imshow(char[i], cmap='gray')
    img_1 = Image.fromarray(char[i])
    img_1 = img_1.convert("RGB")
    
    # Nhận diện ký tự từ Pytesseract
    detected_char = pytesseract.image_to_string(img_1, lang='vie', config='--psm 6')
    
    # Kiểm tra nếu ký tự không rỗng trước khi thêm vào chuỗi txt
    if detected_char:
        txt += detected_char[0]  # Lấy ký tự đầu tiên nếu có
    
    plt.axis('off')

print( txt)




