import cv2
import glob
import numpy as np

index = {
    'train': np.arange(60),
    'val': np.arange(5),
    'test': [0, 1, 2, 3]
}

for _ in ['train', 'test', 'val']:
    for i in index[_]:
        filelist = glob.glob(f'X_{_}/{i}/*.jpg')
        print(f'{_} {i} ...')
        for x in filelist:
            img = cv2.imread(x)
            img = cv2.flip(img, -1)
            cv2.imwrite(x, img)
        # y = np.load(f'y_{_}/{i}.npy')
        # y[:, :, 0] = 639 - y[:, :, 0]
        # y[:, :, 1] = 359 - y[:, :, 1]
        # np.save(f'y_{_}/{i}.npy', y)
