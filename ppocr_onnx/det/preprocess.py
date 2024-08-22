import cv2
import numpy as np


class Resize:
    # Class này để giảm kích thước ảnh để đảm bảo kích thước lớn nhất của ảnh không vượt quá limit_side_len, 
    # giữ nguyên tỷ lệ khung hình mà không bị biến dạng
    def __init__(self, limit_side_len=960):
        self.limit_side_len = limit_side_len

    def __call__(self, img):
        # shape là HWC
        src_h, src_w = img.shape[:2]
        ratio = 1.
        # Nếu vượt quá limit_side_len thì sẽ resize
        if max(src_h, src_w) > self.limit_side_len:
            ratio = self.limit_side_len / src_h if src_h > src_w else self.limit_side_len / src_w
            img = cv2.resize(img, (round(ratio * src_w), round(ratio * src_h)))

        return {
            # Thêm viền đệm màu đen để hình ảnh có kích thước 960x960 pixel
            'image': cv2.copyMakeBorder(
                img,
                0, self.limit_side_len - img.shape[0], 0, self.limit_side_len - img.shape[1],
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            ),
            # Trả về kích thước gốc và tỷ lệ thay đổi ratio
            'shape': np.array([src_h, src_w, ratio]),
        }

class Normalize:
    # Normalize ảnh, ảnh truyền vào có shape HWC
    def __init__(self, mean, std, scale=1/255):
        self.scale = np.float32(scale)
        shape = (1, 1, 3) # ảnh HWC
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')
    def __call__(self, data):
        data['image'] = (data['image'].astype('float32') * self.scale - self.mean) / self.std
        return data

# Tìm ToCHWImage
class HWCToCHW:
    def __call__(self, data):
        data['image'] = data['image'].transpose((2, 0, 1))
        return data

# Lọc key
class PickKeys:
    def __init__(self, *keys):
        self.keys = keys

    def __call__(self, data):
        return [data[key] for key in self.keys]
