import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import pytesseract               
from PIL import Image
plt.style.use('dark_background')
with open('mabiencactinh.json', 'r', encoding='utf-8') as f:
    provinces = json.load(f)





