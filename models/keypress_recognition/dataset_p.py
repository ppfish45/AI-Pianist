import os
import cv2
import glob
import random
import numpy as np
import matplotlib.pyplot as plt

from ipywidgets import IntProgress
from IPython.display import display

path = {
    'X_train': 'data/X_train',
    'X_test': 'data/X_test',
    'X_val': 'data/X_val',
    'y_train': 'data/y_train',
    'y_test': 'data/y_test',
    'y_val': 'data/y_val',
}

X_path = dict()
X_image = dict()
y_org = dict()

X = {
    'single': {
        'white': dict(),
        'black': dict()
    },
    'bundle': {
        'white': dict(),
        'black': dict()
    }
}
y = {
    'white': dict(),
    'black': dict()
}

# X['single']['white']['train']
# y['single']['white']['train']

loaded = False

black_mask = np.array(
    [1, 4, 6, 9, 11, 13, 16, 18, 21, 23, 25, 28, 30, 33, 35, 37, 40, 42, 45, 47, 49, 52, 54, 57, 59, 61, 64,
     66, 69, 71, 73, 76, 78, 81, 83, 85])
white_mask = np.array(
    [0, 2, 3, 5, 7, 8, 10, 12, 14, 15, 17, 19, 20, 22, 24, 26, 27, 29, 31, 32, 34, 36, 38, 39, 41, 43, 44, 46, 48,
     50, 51, 53, 55, 56, 58, 60, 62, 63, 65, 67, 68, 70, 72, 74, 75, 77, 79, 80, 82, 84, 86, 87])

def load_all_data(spliter=['train', 'test', 'val'], to_memory=False, shuffle=False, flatten=True):
    global loaded
    for name in spliter:
        # X
        fp = os.path.join(path['X_' + name], '*.jpg')
        filelist = sorted(glob.glob(fp))
        X_path[name] = filelist
        # y
        fp = os.path.join(path['y_' + name], 'y_' + name + '.npy')
        y_org[name] = np.load(fp)
        y_org[name] = y_org[name][:, 21:109]
        # sanity check
        assert len(X_path[name]) == y_org[name].shape[0]
        # shuffle
        if shuffle:
            mask = np.arange(y_org[name].shape[0])
            random.shuffle(mask)
            X_path[name] = np.array(X_path[name])[mask]
            y_org[name] = y_org[name][mask]
        # load to memory
        if to_memory:
            loaded = True
            X_image[name] = []
            bar = IntProgress(max=len(X_path[name]))
            display(bar)
            for p in X_path[name]:
                img = cv2.imread(p)
                X_image[name].append(img)
                bar.value += 1
            bar.close()
        print(name + ' data loading finished ...')
        print('')
    seperate(spliter, flatten=flatten)

def get_white_keys(img, paddings=0):
    height, width, _ = img.shape # HWC
    unit_width = width // 52
    left = np.arange(52) * unit_width + paddings
    right = (np.arange(52) + 1) * unit_width + paddings
    # add paddings to original img
    padding = np.zeros([height, paddings, 3])
    img = np.concatenate((padding, img, padding), axis=1)
    keys = []
    for i in range(52):
        keys.append(img[:, left[i] - paddings : right[i] + paddings, :])
    keys = np.array(keys, dtype=np.uint8)
    return keys

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

def get_black_keys(img, boundaries, paddings=0):
    height, width, _ = img.shape # HWC
    left = boundaries[:, 0] + paddings
    right = boundaries[:, 1] + paddings
    # add paddings to original img
    padding = np.zeros([height, paddings, 3])
    img = np.concatenate((padding, img, padding), axis=1)
    keys = []
    for i in range(36):
        keys.append(img[:, left[i] - paddings : right[i] + paddings + 1, :])
    keys = np.array(keys, dtype=np.uint8)
    return keys

def seperate(spliter=['train', 'test', 'val'], flatten=True):
    if not loaded:
        print('Please make sure all images are loaded into momory. Use load_all_data(to_memory=True).')
        return

    single_paddings = 2
    bundle_paddings = 10
    white_single_width = 17 + 2 * single_paddings
    white_bundle_width = 17 + 2 * bundle_paddings
    black_single_width = 16 + 2 * single_paddings
    black_bundle_width = 16 + 2 * bundle_paddings
    height = 106
    width = 884

    print('Start seperating keyboard ...')
    print(f'  White single width: {white_single_width}px')
    print(f'  Black single width: {black_single_width}px')
    print(f'  White bundle width: {white_bundle_width}px')
    print(f'  Black bundle width: {black_bundle_width}px')

    for name in spliter:
        black_coor = None
        for img in X_image[name]:
            black_coor = get_black_boundaries(img)
            if len(black_coor) == 36:
                break
        X['single']['white'][name] = []
        X['single']['black'][name] = []
        X['bundle']['white'][name] = []
        X['bundle']['black'][name] = []
        y['white'][name] = []
        y['black'][name] = []

        bar = IntProgress(max=len(X_image[name]))
        display(bar)

        for i, img in enumerate(X_image[name]):
            new_white_single = get_white_keys(img, paddings=single_paddings)
            new_black_single = get_black_keys(img, black_coor, paddings=single_paddings)
            new_white_bundle = get_white_keys(img, paddings=bundle_paddings)
            new_black_bundle = get_black_keys(img, black_coor, paddings=bundle_paddings)
            X['single']['white'][name].append(new_white_single)
            X['single']['black'][name].append(new_black_single)
            X['bundle']['white'][name].append(new_white_bundle)
            X['bundle']['black'][name].append(new_black_bundle)
            y['white'][name].append(y_org[name][i][white_mask])
            y['black'][name].append(y_org[name][i][black_mask])
            bar.value += 1
        bar.close()
        
        print('In ' + name + 'set: ')
        for kind in ['white', 'black']:
            for k2 in ['single', 'bundle']:
                X[k2][kind][name] = np.array(X[k2][kind][name])
                if flatten:
                    shape = X[k2][kind][name].shape
                    X[k2][kind][name] = X[k2][kind][name].reshape([-1, shape[-3], shape[-2], shape[-1]])
            y[kind][name] = np.array(y[kind][name])
            if flatten:
                y[kind][name] = y[kind][name].flatten()
            print('  # of pressed ' + kind + ' key: ' + str(np.sum(y[kind][name] > 0)))
            print('  # of unpressed ' + kind + ' key: ' + str(np.sum(y[kind][name] <= 0)))
    
class data_batch:
    def __init__(
        self,
        type='train',
        size='single',
        color='white',
        batch_size=64,
        need_velocity=True,
        NCHW=True,
        max_num=-1
    ):
        self.size = size
        self.type = type
        self.color = color
        self.batch_size = batch_size
        self.NCHW = NCHW
        self.max_num = max_num
        self.pressed = []
        self.unpressed = []
        self.num_pressed = 0
        self.num_unpressed = 0
        self.need_velocity = need_velocity
        for i, x in enumerate(y[color][type]):
            if x > 0:
                self.pressed.append(i)
                self.num_pressed += 1
            else:
                self.unpressed.append(i)
                self.num_unpressed += 1
        if self.max_num == -1:
            self.max_num = len(self.unpressed)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        start = self.index * self.batch_size // 2
        end = (self.index + 1) * self.batch_size // 2
        if start >= self.max_num:
            raise StopIteration
        if end >= self.max_num:
            end = self.max_num
            start = end - self.batch_size // 2
        ind = []
        s = start % self.num_pressed
        t = end % self.num_pressed
        if start // self.num_pressed == end // self.num_pressed:
            ind.append(self.pressed[s:t])
        else:
            ind.append(self.pressed[s:])
            ind.append(self.pressed[:t])
        ind.append(self.unpressed[start:end])
        ind = np.array(ind).flatten()
        X_return = X[self.size][self.color][self.type][ind]
        if self.NCHW:
            X_return = np.transpose(X_return, (0, 3, 1, 2))
        y_return = y[self.color][self.type][ind]
        if not self.need_velocity:
            y_return = (y_return > 0).astype(np.int)
        return (X_return, y_return)
        