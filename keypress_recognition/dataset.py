import os
import warnings
import cv2
import random
import numpy as np
from keypress_recognition import separate

path = {
    'X_train': 'keypress_recognition/dataset/X_train',
    'X_test': 'keypress_recognition/dataset/X_test',
    'X_val': 'keypress_recognition/dataset/X_val',
    'y_train': 'keypress_recognition/dataset/y_train/y_train.npy',
    'y_test': 'keypress_recognition/dataset/y_test/y_test.npy',
    'y_val': 'keypress_recognition/dataset/y_val/y_val.npy'
}

def is_jpg(fp):
    return os.path.splitext(fp)[1] == '.jpg'

X_path = dict()
y = dict()

black_mask = np.array(
    [1, 4, 6, 9, 11, 13, 16, 18, 21, 23, 25, 28, 30, 33, 35, 37, 40, 42, 45, 47, 49, 52, 54, 57, 59, 61, 64,
     66, 69, 71, 73, 76, 78, 81, 83, 85])
white_mask = np.array(
    [0, 2, 3, 5, 7, 8, 10, 12, 14, 15, 17, 19, 20, 22, 24, 26, 27, 29, 31, 32, 34, 36, 38, 39, 41, 43, 44, 46, 48,
     50, 51, 53, 55, 56, 58, 60, 62, 63, 65, 67, 68, 70, 72, 74, 75, 77, 79, 80, 82, 84, 86, 87])


# convert completed
def load_all_data(**kwargs):
    for name, p in path.items():
        if name[0] == 'X':
            X_path[name] = [os.path.join(p, filepath) for filepath in os.listdir(p) if is_jpg(filepath)]
            if len(X_path[name]) == 0:
                warnings.warn(path[name] + " contains no image files (<number>.jpg). The corresponding data is set to empty list.")
        if name[0] == 'y':
            if os.path.isfile(path[name]):
                y[name] = np.load(path[name]) > 0
            else:
                y[name] = None
                warnings.warn(path[name] + " not found. Setting corresponding data to None.")

    # make sure X and y are perfectly aligned, and shuffle
    for shit in ('train', 'val', 'test'):
        yname = f'y_{shit}'
        xname = f'X_{shit}'
        X_path[xname] = np.array(X_path[xname])
        if y[yname] is not None:
            assert len(X_path[xname]) == y[yname].shape[0], f'In set {shit}: {len(X_path[xname])} and {y[yname].shape}[0] do not match'
            mask = np.arange(len(X_path[xname]))
            np.random.shuffle(mask)
            if kwargs.get(shit) is not None:
                mask = mask[:kwargs[shit]]
                X_path[xname] = X_path[xname][mask]
                y[yname] = y[yname][mask]

    for x in X_path:
        print('# of ' + x + ': ' + str(len(X_path[x])))


# convert completed
def get_sample(type='train', img=True, method=0):
    """
    :param type: train | val | test
    :param img: a image indicator
    :param method:
        0 -- unbundled separation
        1 -- bundled separation
        2 -- no separation
    :return: image, notes of size (88, )
    """
    # random frame selection index
    x_ = X_path[f'X_{type}']
    y_ = y[f'y_{type}']
    idx = random.randint(0, len(x_) - 1)  
    path = x_[idx]
    notes = y_[idx]

    # preprocessing
    notes = notes[21:109]
    if img:
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        if method == 0:
            white, black = separate.separate(image)
        elif method == 1:
            white, black = separate.separate(image, bundle=True)
        else:
            return image, notes

        return white, black, notes[white_mask], notes[black_mask]
    else:
        return path, notes


def get_num_of_data(type='train'):
    return X_path[f'X_{type}'].shape[0]


def show_corresponding_label(type='train', index=0):
    # return an np array of size (frame_num,128), representing the note occurrence of that video
    return y[f'y_{type}'][index][1]


'''
Image format: NCHW
Batch format: white, black
    return_1 -- [batch_size, file_path] img
    return_2 -- [batch_size, 88] corresponding note
    
    or 
    
    black/white -- [batch_size * 36/54, 3, 106, 12]
    
'''


class data_batch:
    def __init__(self, type='train', method=2, batch_size=64, NCHW=True, max_num=-1):
        self.type = type
        self.batch_size = batch_size
        self.NCHW = NCHW
        self.max_num = max_num
        if self.max_num == -1:
            self.max_num = get_num_of_data(self.type)
        self.max_num = min(self.max_num, get_num_of_data(self.type))
        self.method = method

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        start = self.index * self.batch_size
        end = (self.index + 1) * self.batch_size
        if start >= self.max_num:
            raise StopIteration
        if end >= self.max_num:
            end = self.max_num
            start = end - self.batch_size

        if self.method == 0:
            white = np.empty((0, separate.white_key_height, separate.white_key_width, 3))
            black = np.empty((0, separate.black_key_height, separate.black_key_width, 3))

        elif self.method == 1:
            white = np.empty((0, separate.white_key_height, separate.white_key_width_bundle, 3))
            black = np.empty((0, separate.black_key_height, separate.black_key_width_bundle, 3))

        if self.method == 0:
            for x in X_path[f'X_{self.type}'][start: end]:
                w, b = separate.separate(cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB))
                white = np.concatenate((white, np.array(w)), axis=0)
                black = np.concatenate((black, np.array(b)), axis=0)
        elif self.method == 1:
            for x in X_path[f'X_{self.type}'][start: end]:
                w, b = separate.separate(cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB), bundle=True)
                white = np.concatenate((white, np.array(w)), axis=0)
                black = np.concatenate((black, np.array(b)), axis=0)
        else:
            X_return = np.array(
                [cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB) for x in X_path[f'X_{self.type}'][start: end]])

        if self.NCHW and self.method == 2:
            X_return = np.array(np.transpose(X_return, (0, 3, 1, 2)))  # convert to NCHW
        elif self.NCHW and (self.method == 0 or self.method == 1):
            white = np.array(np.transpose(white, (0, 3, 1, 2)))
            black = np.array(np.transpose(black, (0, 3, 1, 2)))
        y_return = y[f'y_{self.type}'][start: end]
        y_return = y_return[:, 21:109]

        self.index += 1
        if self.method == 2:
            return np.array(X_return), y_return
        else:
            return white, black, y_return[:, white_mask].flatten(), y_return[:, black_mask].flatten()


# load_all_data()
