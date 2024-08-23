"""
There are utilities, some of them are useful, but the other ones useless ._.
I'm too lazy to get rid of useless functions
"""
import re
import os
import cv2
import time
import torch
import numpy as np
from datetime import datetime

BGR_COLORS = {
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "amber": (0, 191, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0)
}
VEHICLES = {
    'vi': ['xe buyt', 'o to', 'o to', 'xe may', 'xe tai'],
    'en': ['bike', 'bus', 'car', 'motorbike', 'truck'],
    'es': ['bicicleta', 'autobus', 'auto', 'moto', 'camion'],
    'fr': ['velo', 'bus', 'voiture', 'moto', 'camion'],
    'coco': ['bus', 'car', 'motorcycle', 'truck', 'bicycle'],
    'coco_vi': ['xe buyt', 'o to', 'xe may', 'xe tai', 'xe dap']
}
COLOR_PALETTE = np.random.uniform(0, 255, size=(len(VEHICLES['en']), 3))

COLOR_HSV = {
        'black': [(0, 0, 0), (180, 255, 30)],
        'blue': [(100, 150, 0), (140, 255, 255)],
        'brown': [(10, 100, 20), (20, 255, 200)],  # Brown color range
        'gray': [(0, 0, 50), (180, 50, 200)],      # Lighter gray color range
        'orange': [(10, 100, 20), (25, 255, 255)], # Orange color range
        'pink': [(140, 50, 50), (170, 255, 255)],  # Pink color range
        'purple': [(125, 50, 50), (155, 255, 255)],# Purple color range
        'red': [(0, 70, 50), (10, 255, 255)],      # Red color range
        'white': [(0, 0, 200), (180, 20, 255)],    # White color range
        'yellow': [(20, 100, 100), (30, 255, 255)] # Yellow color range
    }


class MyDict(dict):
    def __getattribute__(self, item):
        return self[item]
    
class Vehicle:
    track_id: int = 0
    vehicle_detection_score: float = 0.0
    vehicle_type: str = ""
    vehicle_bbox: np.ndarray = None # xyxy
    vehicle_image: np.ndarray = None
    license_plate_image: np.ndarray = None
    license_plate: str = ""
    license_plate_bbox: np.ndarray = None # xyxy
    license_plate_score: float = 0.0
    tracking_feature_vector: np.ndarray
    is_recognized: bool = False

def draw_detections(img, box, class_id, lang='en'):
    """
    Draws bounding boxes and labels on the input image based on the detected objects.

    Args:
        img: The input image to draw detections on.
        box: Detected bounding box.
        score: Corresponding detection score.
        class_id: Class ID for the detected object.

    Returns:
        None
    """
    x1, y1, w, h = box[0], box[1], box[2], box[3]
    color = COLOR_PALETTE[class_id]
    cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
    label = f'{map_label(class_id, VEHICLES[lang])}'
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_x = x1
    label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
    cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
                    cv2.FILLED)
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img

def gettime():
    return time.time()


def map_label(class_idx, vehicle_labels):
    """
    Map argmax output to label name following COCO Object
    """
    return vehicle_labels[class_idx]


def check_image_size(image, w_thres, h_thres):
    """
    Ignore small images
    Args: image, w_thres, h_thres
    """
    if w_thres is None:
        w_thres = 64
    if h_thres is None:
        h_thres = 64
    width, height, _ = image.shape
    if (width >= w_thres) and (height >= h_thres):
        return True
    else:
        return False


def draw_text(img, text,
              pos=(0, 0),
              font=cv2.FONT_HERSHEY_SIMPLEX,
              font_scale=1,
              font_thickness=2,
              text_color=(255, 255, 255),
              text_color_bg=(0, 0, 0)):
    """
    Minor modification of cv2.putText to add background color
    """
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    bg_h = int(text_h * 1.05)  # expand the background height a bit
    cv2.rectangle(img, pos, (x + text_w, y + bg_h), text_color_bg, -1)
    pos = (x, y + text_h + font_scale)
    cv2.putText(img, text, pos, font, font_scale, text_color, font_thickness, cv2.LINE_AA)


def draw_box(img, pt1, pt2, color, thickness, r, d):
    """
    Draw more fancy bounding box
    """
    x1, y1 = pt1
    x2, y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def draw_tracked_boxes(img, bbox, identities=None, offset=(0, 0)):
    """
    Draw box tracked by deepsort
    """
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        idx = int(identities[i]) if identities is not None else 0
        color = compute_color(idx)
        label = '{}{:d}'.format("", idx)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def compute_color(label):
    """
    Add borders of different colors
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def set_hd_resolution(image):
    """
    Set video resolution (for displaying only)
    Arg:
        image (OpenCV image): video frame read by cv2
    """
    height, width, _ = image.shape
    ratio = height / width
    image = cv2.resize(image, (1280, int(1280 * ratio)))
    return image


def resize_(image, scale):
    """
    Compress cv2 resize function
    """
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dsize = (width, height)
    image = cv2.resize(image, dsize)
    image = cv2.copyMakeBorder(src=image, top=5, bottom=5, left=5, right=5,
                               value=[0, 255, 0],
                               borderType=cv2.BORDER_CONSTANT)
    return image


def delete_file(path):
    """
    Delete generated file during inference
    """
    if os.path.exists(path):
        os.remove(path)


def preprocess_detection(detection):
    """
    Process yolov8 output
    """
    bboxes = detection[0].boxes
    xywhs = bboxes.xywh
    confss = bboxes.conf.unsqueeze(1)
    classes = bboxes.cls.unsqueeze(1)
    return torch.cat((xywhs, confss, classes), 1).cpu()


def argmax(listing):
    """
    Find the index of the maximum value in a list
    """
    return np.argmax(listing)


def argmin(listing):
    """
    Find the index of the minimum value in a list
    """
    return np.argmin(listing)


def get_time_now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def crop_expanded_plate(plate_xyxy, cropped_vehicle, expand_ratio=0.2):
    """
    Crops an expanded area around the given coordinates in the image.

    Args:
    plate_xyxy (tuple): A tuple containing the coordinates (x_min, y_min, x_max, y_max) of the plate.
    cropped_vehicle (numpy.ndarray): The image from which the plate is to be cropped.
    expand_ratio (float): The ratio by which to expand the cropping area on each side. Default is 0.1 (10%).

    Returns:
    numpy.ndarray: The cropped image of the expanded plate.
    """
    # Original coordinates
    x_min, y_min, x_max, y_max = plate_xyxy

    # Calculate the width and height of the original cropping area
    width = x_max - x_min
    height = y_max - y_min

    # Calculate the expansion amount (10% of the width and height by default)
    expand_x = int(expand_ratio * width)
    expand_y = int(expand_ratio * height)

    # Calculate the new coordinates with expansion
    new_x_min = max(x_min - expand_x, 0)
    new_y_min = max(y_min - expand_y, 0)
    
    # (height, width, 3) là shape của ảnh
    new_x_max = min(x_max + expand_x, cropped_vehicle.shape[1])
    new_y_max = min(y_max + expand_y, cropped_vehicle.shape[0])

    # Crop the expanded area
    cropped_plate = cropped_vehicle[new_y_min:new_y_max, new_x_min:new_x_max, :]

    return cropped_plate
    
def check_legit_plate(s):
    # Remove unwanted characters
    s_cleaned = re.sub(r'[.\-\s]', '', s)
    
    # thiếu ký tự: recog lại
    if len(s_cleaned) > 9 or len(s_cleaned) < 8:
        return False
    return True
    
    # # chứa ký tự viết thường: recog lại
    # if re.search(r'[a-z]', s):
    #     return False
    
    # return True

    # =============================
    # Regular expressions for different cases
    # biển quân đội
    # pattern1 = r'^[A-Z]{2}[0-9]{4}$'  # Matches exactly 2 letters followed by exactly 4 digits
    # # biển khác
    # pattern2 = r'[A-Z][0-9]{4,}'      # Matches an alphabet character followed by at least 4 digits
    
    # # Check if the cleaned string matches either pattern
    # if re.search(pattern1, s_cleaned) or (re.search(pattern2, s_cleaned) and not re.match(r'^[A-Z]{2}', s_cleaned)):
    #     return True
    # else:
    #     return False
    # ==========================



def correct_plate(s):

    dict_char_to_int = {
                        'A': '4', 
                        'B': '8',
                        'C': '0', # 61-B19997C
                        'D': '0', # nếu sau chữ cái cuối cùng có 5 
                        'G': '6', # G1Z1112 (s)
                        'U': '0', # 6UC54536 (ds n)
                        'I': '1', # SIF69116
                        'T': '1', # 59-YT14999 (ds n)
                        'L': '4', # 63BL25751, 6L.3Z41204, 85-R3L336, L8-E11362, 5L-X85917
                        'S': '5', #51-SS4867 (ds n), 89-HS3589 (ds n), SIF69116 (ds n) 
                        'Z': '2', # 61-SZ2916 (ds n), 6L.3Z41204 (ds n), 67LZ12199 (ds n)
                        # 'Z': '7', # 63-1Z9710, Z1-B4054.00 (deepsort k legit)
                        }
    dict_int_to_char = {
                        # '0': 'C', # 60-0192160 (ds n)
                        '0': 'D', # 61-01663.65, 61LD036.81(TH đúng), 77-01147.31
                        '2': 'Z',
                        '1': 'L', # 37-1217020, 78-119281, 371217020
                        # '1': 'T', # 47-11039043 (ds n)
                        # '1': 'X', # 59-1287267, 63-1Z9710, 61121112
                        '6': 'G', # 516153602
                        '8': 'B', # 48-81583.65, 72803844 (rất nhiều)
                        } 
    dict_int_to_int = {'0': '8', # 09-H53589, 06-B10686            
                        }
    # Ký tự thừa: 71-C2K46290 (k) (ds n), 543F24992 (ds n) (3)
    # thiếu ký tự: 1-C26290, 9-X2979, 37-L2170, 60A2407, 61A3407, G1Z1112
    # sai: 36-D1520b, 64.d241204, F116436
    # bỏ ký tự đặc biệt -> chuẩn hóa lại biển số -> Nếu tồn tại biển số rồi -> bỏ qua
    
    s = re.sub(r'[.\-\s]', '', s)
    # case 1: số 0 ở đầu
    if s[0] == '0':
        s = '8' + s[1:]
        
    # case 2: 2 ký tự đầu bắt buộc là số, nếu dính chữ thì sửa lại
    # Kiểm tra và thay thế các ký tự đầu tiên
    if s[0].isupper() or s[1].isupper():
        s = ''.join(dict_char_to_int.get(c, c) for c in s[:2]) + s[2:]
        
    # case 3: 2 ký tự tiếp theo
    # 2 ký tự là chữ
    s_2_4 = s[2:4]
    if s_2_4[0].isupper() and s_2_4[1].isupper():  # Nếu cả hai ký tự đều là chữ viết hoa
        if len(s) == 8:  # Trường hợp 1: len(s) == 8
            s = s[:3] + dict_char_to_int.get(s[3], s[3]) + s[4:]
        # Trường hợp 2: len(s) == 9 -> Không làm gì cả
    # 2 ký tự là số:
    elif s_2_4.isdigit():
        s = s[:2] + dict_int_to_char.get(s[2], s[2]) + s[3:]
    # 1 ký tự là chữ, 1 ký tự là số
    elif s_2_4.isupper():  # Trường hợp 1: số trước chữ
        s = s[:2] + dict_int_to_char.get(s[2], s[2]) + dict_char_to_int.get(s[3], s[3]) + s[4:]
    else: s = s
    
    # case 4: các ký tự còn lại phải là số:
    s_4 = s[4:]
    def replace(match):
        char = match.group(0)
        return dict_char_to_int.get(char, char)
    # Tìm tất cả ký tự viết hoa và thay thế chúng
    result = re.sub(r'[A-Z]', replace, s_4)
    s = s[:4] + result 
    
    return s

# print(correct_plate('LB-6T1Z63D'))
# print(check_legit_plate('LB-6T1Z63D'))
    
    