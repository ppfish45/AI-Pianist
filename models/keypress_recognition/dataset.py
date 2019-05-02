import os
import cv2
import glob
import random
import numpy as np

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
                filelist = glob.glob(os.path.join(f, '*.jpg'))
                filelist = sorted(filelist, key=lambda x: int(x.split('/')[-1].split('.')[0]))
                # print('Load ' + f + ' ...')
                for file in filelist:
                    X_path[name].append(file)
        if name[0] == 'y':
            y[name] = np.empty([0, 128])
            # filter .DS_Store out
            files = sorted([x for x in os.listdir(p) if x[0] != '.'], key=lambda x: int(x.split('.')[0]))
            files = [os.path.join(p, x) for x in files]
            for f in files:
                # print('Load ' + f + ' ...')
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


# newly added func
def show_corresponding_label(type='train', index=0):
    # return an np array of size (128,), representing the note occurrence of that frame
    return y[f'y_{type}'][index]


# convert completed
def get_sample(type='train', img=True, dir=0, img_size=(334, 40)):
    ind = random.randint(0, len(X_path[f'X_{type}']))  # random frame selection index
    path = X_path[f'X_{type}'][ind]
    notes = y[f'y_{type}'][ind]
    if img:
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, img_size, interpolation=cv2.INTER_CUBIC)
        return image, notes
    else:
        return path, notes


# stay the same
def get_num_of_data(type='train'):
    return X_path[f'X_{type}'].shape[0]


'''
Image format: NCHW
Batch format:
    return_1 -- [batch_size, file_path] img
    return_2 -- [batch_size, 128] corresponding note
    
'''


class data_batch:
    def __init__(self, type='train', batch_size=64, image_size=(224, 224), NCHW=True, max_num=-1, random_dir=False):
        self.type = type
        self.batch_size = batch_size
        self.image_size = image_size
        self.NCHW = NCHW
        self.max_num = max_num
        if self.max_num == -1:
            self.max_num = get_num_of_data(self.type)
        self.max_num = min(self.max_num, get_num_of_data(self.type))
        self.random_dir = random_dir

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
                    for x in X_path[f'X_{self.type}'][start: end]]
        if self.NCHW:
            X_return = np.array(np.transpose(X_return, (0, 3, 1, 2)))  # convert to NCHW
        y_return = y[f'y_{self.type}'][start: end]

        self.index += 1
        return np.array(X_return), y_return


load_all_data()
