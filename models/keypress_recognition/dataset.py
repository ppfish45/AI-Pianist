import os
import cv2
import glob
import random
import numpy as np
from ..keypress_recognition import separate

path = {
    'K_train': 'dataset/K_train',
    'K_test': 'dataset/K_test',
    'K_val': 'dataset/K_val',
    'y_train': 'dataset/y_train',
    'y_test': 'dataset/y_test',
    'y_val': 'dataset/y_val'
}

X_path = dict()
y = dict()

black_mask = [1, 4, 6, 9, 11, 13, 16, 18, 21, 23, 25, 28, 30, 33, 35, 37, 40, 42, 45, 47, 49, 52, 54, 57, 59, 61, 64,
              66, 69, 71, 73, 76, 78, 81, 83, 85]
white_mask = [0, 2, 3, 5, 7, 8, 10, 12, 14, 15, 17, 19, 20, 22, 24, 26, 27, 31, 32, 34, 36, 38, 39, 41, 43, 44, 46, 48,
              50, 51, 53, 55, 56, 58, 60, 62, 63, 65, 67, 68, 70, 72, 74, 75, 77, 79, 80, 82, 84, 86, 87, 88]

# convert completed
def load_all_data():
    print(white_mask.__len__())
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
    idx = random.randint(0, len(X_path[f'X_{type}']))  # random frame selection index
    path = X_path[f'X_{type}'][idx]
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
    return X_path[f'X_{type}'].shape[0]


def show_corresponding_label(type='train', index=0):
    # return an np array of size (frame_num,128), representing the note occurrence of that video
    return y[f'y_{type}'][index][1]


'''
Image format: NCHW
Batch format: white, black
    return_1 -- [batch_size, file_path] img
    return_2 -- [batch_size, 88] corresponding note
    
'''


class data_batch:
    def __init__(self, type='train', method=0, batch_size=64, image_size=(224, 224), NCHW=True, max_num=-1):
        self.type = type
        self.batch_size = batch_size
        self.image_size = image_size
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
        for x in X_path[f'X_{self.type}'][start: end]:
            if self.method == 0:
                separate.separate(x)
            elif self.method == 1:
                separate.separate(x, bundle=True)
        X_return = [cv2.resize(cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB), self.image_size, interpolation=cv2.INTER_CUBIC)]

        if self.NCHW:
            X_return = np.array(np.transpose(X_return, (0, 3, 1, 2)))  # convert to NCHW
        y_return = y[f'y_{self.type}'][start: end]

        self.index += 1
        return np.array(X_return), y_return[20:108]


if __name__ == "__main__":
    load_all_data()
