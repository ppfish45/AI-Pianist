import os
import cv2
import glob
import random
import numpy as np
from . import separate

path = {
    'K_train': 'keypress_recognition/dataset/K_train',
    'K_test': 'keypress_recognition/dataset/K_test',
    'K_val': 'keypress_recognition/dataset/K_val',
    'y_train': 'keypress_recognition/dataset/y_train',
    'y_test': 'keypress_recognition/dataset/y_test',
    'y_val': 'keypress_recognition/dataset/y_val'
}

X_path = dict()
y = dict()

black_mask = np.array(
    [1, 4, 6, 9, 11, 13, 16, 18, 21, 23, 25, 28, 30, 33, 35, 37, 40, 42, 45, 47, 49, 52, 54, 57, 59, 61, 64,
     66, 69, 71, 73, 76, 78, 81, 83, 85])
white_mask = np.array(
    [0, 2, 3, 5, 7, 8, 10, 12, 14, 15, 17, 19, 20, 22, 24, 26, 27, 31, 32, 34, 36, 38, 39, 41, 43, 44, 46, 48,
     50, 51, 53, 55, 56, 58, 60, 62, 63, 65, 67, 68, 70, 72, 74, 75, 77, 79, 80, 82, 84, 86, 87])


# convert completed
def load_all_data():
    for name in path:
        p = path[name]
        if name[0] == 'K':
            X_path[name] = []
            # filter .DS_Store out
            folders = sorted([x for x in os.listdir(p) if x[0] != '.'], key=lambda x: int(x))
            folders = [os.path.join(p, x) for x in folders]
            for f in folders:
                fileList = glob.glob(os.path.join(f, '*.jpg'))
                fileList = sorted(fileList, key=lambda x: int(x.split('/')[-1].split('.')[0]))
                # print('Load ' + f + ' ...')
                for file in fileList:
                    X_path[name].append(file)
        if name[0] == 'y':
            y[name] = np.empty([0, 128])
            # filter .DS_Store out
            files = sorted([x for x in os.listdir(p) if x[0] != '.'], key=lambda x: int(x.split('.')[0]))
            files = [os.path.join(p, x) for x in files]
            for f in files:
                # print('Load ' + f + ' ...')
                # np.load(f) of size (total_frame, 128)
                y[name] = np.concatenate((y[name], np.load(f)), axis=0)

    # make sure X and y are perfectly aligned
    assert len(X_path['K_train']) == y['y_train'].shape[0]
    assert len(X_path['K_test']) == y['y_test'].shape[0]
    assert len(X_path['K_val']) == y['y_val'].shape[0]

    for _ in ['train', 'test', 'val']:
        mask = np.arange(len(X_path[f'K_{_}']))
        random.shuffle(mask)
        X_path[f'K_{_}'] = np.array(X_path[f'K_{_}'])
        X_path[f'K_{_}'] = X_path[f'K_{_}'][mask]
        y[f'y_{_}'] = y[f'y_{_}'][mask]

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
    idx = random.randint(0, len(X_path[f'K_{type}']))  # random frame selection index
    path = X_path[f'K_{type}'][idx]
    notes = y[f'y_{type}'][idx]
    notes = notes[20:108]
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
    return X_path[f'K_{type}'].shape[0]


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
            for x in X_path[f'K_{self.type}'][start: end]:
                w, b = separate.separate(cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB))
                white = np.concatenate((white, np.array(w)), axis=0)
                black = np.concatenate((black, np.array(b)), axis=0)
        elif self.method == 1:
            for x in X_path[f'K_{self.type}'][start: end]:
                w, b = separate.separate(cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB), bundle=True)
                print(np.array(b).shape)
                white = np.concatenate((white, np.array(w)), axis=0)
                black = np.concatenate((black, np.array(b)), axis=0)
        else:
            X_return = np.array(
                [cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB) for x in X_path[f'K_{self.type}'][start: end]])
        if self.NCHW and self.method == 2:
            X_return = np.array(np.transpose(X_return, (0, 3, 1, 2)))  # convert to NCHW
        if self.NCHW and (self.method == 0 or self.method == 1):
            white = np.array(np.transpose(white, (0, 3, 1, 2)))
            black = np.array(np.transpose(black, (0, 3, 1, 2)))
        y_return = y[f'y_{self.type}'][start: end]
        y_return = y_return[:, 20:108]

        self.index += 1
        if self.method == 2:
            return np.array(X_return), y_return
        else:
            return white, black, y_return[:, white_mask], y_return[:, black_mask]


load_all_data()
