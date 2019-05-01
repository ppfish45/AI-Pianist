import os
import cv2
import copy
import glob
import random
import numpy as np
from matplotlib.image import imread

path = {
    'X_train': 'dataset/X_train',
    'X_test': 'dataset/X_test',
    'X_val': 'dataset/X_val',
    'y_train': 'dataset/y_train',
    'y_test': 'dataset/y_test',
    'y_val': 'dataset/y_val'
}

X_path = dict()
y = dict()

def load_all_data():

    for name in path:
        p = path[name]
        if name[0] == 'X':
            X_path[name] = []
            # filter .DS_Store out
            folders = sorted([x for x in os.listdir(p) if x[0] != '.'], key=lambda x: int(x))
            folders = [os.path.join(p, x) for x in folders]
            for f in folders:
                filelist = glob.glob(os.path.join(f, '*.jpg'))
                filelist = sorted(filelist, key=lambda x: int(x.split('/')[-1].split('.')[0]))
                # print('Load ' + f + ' ...')
                for file in filelist:
                    X_path[name].append(file)
        if name[0] == 'y':
            y[name] = np.empty([0, 4, 2])
            # filter .DS_Store out
            files = sorted([x for x in os.listdir(p) if x[0] != '.'], key=lambda x: int(x.split('.')[0]))
            files = [os.path.join(p, x) for x in files]
            for f in files:
                # print('Load ' + f + ' ...')
                y[name] = np.concatenate((y[name], np.load(f)), axis = 0)

    # make sure X and y are perfectly aligned
    assert len(X_path['X_train']) == y['y_train'].shape[0]
    assert len(X_path['X_test']) == y['y_test'].shape[0]
    assert len(X_path['X_val']) == y['y_val'].shape[0]

    for _ in ['train', 'test', 'val']:
        mask = np.arange(len(X_path[f'X_{_}']))
        random.shuffle(mask)
        X_path[f'X_{_}'] = np.array(X_path[f'X_{_}'])
        X_path[f'X_{_}'] = X_path[f'X_{_}'][mask]
        y[f'y_{_}'] = y[f'y_{_}'][mask]

    for x in X_path:
        print('# of ' + x + ': ' + str(len(X_path[x])))

def rotate_image(img, image_size=(224, 224), dir=0):
    ret = copy.copy(img)
    H = cv2.getRotationMatrix2D((image_size[0] / 2, image_size[1] / 2), -90, 1)
    for i in range(dir):
        ret = cv2.warpAffine(ret, H, (ret.shape[1], ret.shape[0]))
    return ret

def rotate_coordinate(pts, image_size=(224, 224), dir=0):
    ret = copy.copy(pts)
    for i in range(dir):
        ret[:, [0, 1]] = ret[:, [1, 0]]
        ret[:, 0] = image_size[1] - 1 - ret[:, 0]
    return ret
        
def show_labelled_image(img, pts, dir=0):
    if isinstance(img, str):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _img = copy.copy(img)
    _img = rotate_image(_img, dir=dir)
    _pts = rotate_coordinate(pts, dir=dir)
    for k, p in enumerate(_pts):
        cv2.circle(_img, (int(p[0]), int(p[1])), 2, (255 * (k % 3 == 0), 255 * (k % 2 == 0), 255), 2)
    return _img

def get_sample(type='train', img=True, dir=0, img_size=(640, 360)):
    ind = random.randint(0, len(X_path[f'X_{type}']))
    path = X_path[f'X_{type}'][ind]
    pts = y[f'y_{type}'][ind]
    if img:
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, img_size, interpolation=cv2.INTER_CUBIC)
        return (image, pts)
    else:
        return (path, pts)

def get_num_of_data(type='train'):
    return X_path[f'X_{type}'].shape[0]

'''
Image format: NCHW
'''

class data_batch():
    def __init__(self, type='train', batch_size=64, image_size=(224, 224), NCHW=True, max_num=-1, random_dir=False):
        self.type = type
        self.batch_size = batch_size
        self.image_size = image_size
        self.NCHW = NCHW
        self.max_num = max_num
        if self.max_num == -1:
            self.max_num = get_num_of_data(self.type)
        self.max_num = min(self.max_num, get_num_of_data(self.type))

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
        X_return = [cv2.resize(cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB),
                               self.image_size, interpolation=cv2.INTER_CUBIC)
                    for x in X_path[f'X_{self.type}'][start : end]]
        if self.NCHW:
            X_return = np.array(np.transpose(X_return, (0, 3, 1, 2))) # convert to NCHW
        y_return = copy.copy(y[f'y_{self.type}'][start : end])
        # resize image
        y_return *= [self.image_size[0] / 640.0, self.image_size[1] / 360.0]

        # randomly rotate images and coordinates
        if random_dir:
            for i in range(batch_size):
                dir = random.randint(0, 3)
                X_return[i] = rotate_image(X_return[i], dir=dir)
                y_return[i] = rotate_coordinate(y_return[i], dir=dir)

        self.index += 1
        return (np.array(X_return), y_return)

load_all_data()
