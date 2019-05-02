import cv2
import os
import glob
import numpy as np
import random

path = {
    'K_train': 'dataset/K_train',
    'K_test': 'dataset/K_test',
    'K_val': 'dataset/K_val',
    'S_train': 'dataset/S_train',
    'S_test': 'dataset/S_test',
    'S_val': 'dataset/S_val'
}

X_path = dict()
y = dict()


img_width = 884
img_height = 106

white_key_count = 52
black_key_count = 36

white_key_width_strict = 884 // white_key_count # 17
white_key_width_tolerence = 2
white_key_width = white_key_width_tolerence * 2 + white_key_width_strict # 21

white_key_height = img_height

black_key_width_strict = white_key_width_strict // 2 # 8
black_key_width_tolerence = 2
black_key_width = black_key_width_tolerence * 2 + black_key_width_strict # 12

black_key_height = img_height

# assert black_key_height == 69 and black_key_width == 10 and white_key_height == 106 and white_key_width == 17, "Incorrect calculation of key dimentions"

def seperate(img):
    """
    img: An interable, each element is a image file of a keyboard. The
    images should be standardized, i.e. rectangular of size 884, 106

    Crop the images to seperate keys and list out in this order: white keys
    from lowest to highest, black keys from lowest to highest

    return: A list of image files .
    """
    assert img.shape[0] == img_width and img.shape[1] == img_height, f"Image file not of size {img_width}, {img_height}"
    white_imgs = [
        img[max(0, i*white_key_width_strict-white_key_width_tolerence):min(img_width, (i+1)*white_key_width_strict+white_key_width_tolerence), :].copy() 
        for i in range(52)
    ]
    black_imgs = [
        img[max(0, i*black_key_width_strict-black_key_width_tolerence):min(img_width, (i+1)*black_key_width_strict+black_key_width_tolerence), :].copy() 
        for i in [1, 4, 6, 9, 11, 13, 16, 18, 21, 23, 25, 28, 30, 33, 35, 37, 40, 42, 45, 47, 49, 52, 54, 57, 59, 61, 64, 66, 69, 71, 73, 76, 78, 81, 83, 85]
    ]
    return white_imgs, black_imgs

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

    for _ in ['train', 'test', 'val']:
        mask = np.arange(len(X_path[f'K_{_}']))
        random.shuffle(mask)
        X_path[f'K_{_}'] = np.array(X_path[f'K_{_}'])
        X_path[f'K_{_}'] = X_path[f'K_{_}'][mask]

    for x in X_path:
        print('# of ' + x + ': ' + str(len(X_path[x])))
    
    return X_path

X = load_all_data()
for x in X:
    for path in X[x]:
        img = cv2.imread(path)
        img = np.transpose(img, (1, 0, 2))
        white, black = seperate(img)
        white_num = 0
        black_num = 0
        new_path = path.replace('K','S')
        new_path = new_path.replace('.jpg','_')
        for w in white:
            white_num += 1
            cv2.imwrite(f'{new_path}w_{white_num}.jpg', w)
        for b in black:
            black_num += 1
            cv2.imwrite(f'{new_path}b_{black_num}.jpg', b)
