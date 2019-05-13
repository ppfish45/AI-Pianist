import os
import cv2
import numpy as np


if os.getcwd().endswith("models"):
    X_path = 'keypress_recognition/dataset/X_test'
else:
    X_path = 'dataset/X_test'
X_path_list = [os.path.join(X_path, path) for path in os.listdir(X_path) if os.path.splitext(path)[1] == '.jpg']


X = {
    'black': [],
    'white': []
}
single_paddings = 2
bundle_paddings = 10
white_single_width = 17 + 2 * single_paddings
white_bundle_width = 17 + 2 * bundle_paddings
black_single_width = 16 + 2 * single_paddings
black_bundle_width = 16 + 2 * bundle_paddings
height = 106
width = 884


def get_black_boundaries(img, expected_width=16):
    '''
    want a grayscale image, (h, w) format
    '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, bw = cv2.threshold(blur, 127, 255, cv2.THRESH_OTSU)
    upper = np.mean(bw[0:5], axis=0)
    
    black_keys = []
    last = -1

    for i, x in enumerate(upper):
        if x <= 255 / 5:
            if last == -1:
                last = i
        else:
            if last != -1:
                black_keys.append([last, i - 1])
                last = -1

    for i, coor in enumerate(black_keys):
        x, y = coor
        offset = (expected_width - (y - x + 1)) // 2
        coor[0] = x - offset
        coor[1] = y + (expected_width - (y - x + 1) - offset)
    
    return np.array(black_keys)


def get_white_keys(img, paddings):
    height, width, _ = img.shape # HWC
    unit_width = width // 52
    left = np.arange(52) * unit_width + paddings
    right = (np.arange(52) + 1) * unit_width + paddings
    # add paddings to original img
    # padding = np.zeros([height, paddings, 3])
    # img = np.concatenate((padding, img, padding), axis=1).astype(np.uint8)
    img = np.pad(img, ((0, 0), (paddings, paddings), (0, 0)), mode='constant', constant_values=0)
    return np.array([img[:, left[i] - paddings : right[i] + paddings, :] for i in range(52)])


def get_black_keys(img, boundaries, paddings):
    height, width, _ = img.shape # HWC
    left = boundaries[:, 0] + paddings
    right = boundaries[:, 1] + paddings
    # add paddings to original img
    # padding = np.zeros([height, paddings, 3])
    # img = np.concatenate((padding, img, padding), axis=1).astype(np.uint8)
    img = np.pad(img, ((0, 0), (paddings, paddings), (0, 0)), mode='constant', constant_values=0)
    return np.array([img[:, left[i] - paddings : right[i] + paddings + 1, :] for i in range(36)])


black_coor = None
for path in X_path_list:
    img = cv2.imread(path)
    black_coor = get_black_boundaries(img)
    if len(black_coor) == 36:
        break


class data_batch:
    def __init__(self, size, NCHW=True, concatenate=False):
        if size != 'single' and size != 'bundle':
            raise ValueError("Expected 'single' or 'bundle'")
        if concatenate:
            raise NotImplementedError
        self.len = len(X_path_list)
        self.bundle = (size == 'bundle')
        self.NCHW = NCHW
        self.concatenate = concatenate


    def __len__(self):
        return self.len            


    def __iter__(self):
        self.index = 0
        return self


    def __next__(self):
        if self.index >= self.len:
            raise StopIteration
        else:
            img_path = X_path_list[self.index]
            self.index += 1
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        white_keys = get_white_keys(img, (bundle_paddings if self.bundle else single_paddings))
        black_keys = get_black_keys(img, black_coor, (bundle_paddings if self.bundle else single_paddings))
        if self.NCHW:
            white_keys = np.transpose(white_keys, (0, 3, 1, 2))
            black_keys = np.transpose(black_keys, (0, 3, 1, 2))
        return white_keys, black_keys

