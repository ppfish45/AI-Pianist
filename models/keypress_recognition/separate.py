img_width = 884
img_height = 106

white_key_count = 52
black_key_count = 36

white_key_width_strict = 884 // white_key_count  # 17
white_key_width_tolerence = 2
white_key_width = white_key_width_tolerence * 2 + white_key_width_strict  # 21
white_key_width_bundle = 3 * white_key_width_strict  # 51

white_key_height = img_height

black_key_width_strict = white_key_width_strict // 2  # 8
black_key_width_tolerence = 2
black_key_width = black_key_width_tolerence * 2 + black_key_width_strict  # 12
black_key_width_bundle = 3 * black_key_width_strict  # 24

black_key_height = img_height

assert black_key_height == 106 and black_key_width == 12 and white_key_height == 106 and white_key_width == 21, "Incorrect calculation of key dimentions"


def get_bounding_box(img, bundle=False):
    """
    img: An interable, each element is a image file of a keyboard. The
    images should be standardized, i.e. rectangular of size 106, 884

    Get cropping bounding boxes of seperate keys and list out in this order: white keys
    from lowest to highest, black keys from lowest to highest

    return: A list of dimension 4 bounding boxes: left, right, up, down.
    """
    assert img.shape[1] == img_width and img.shape[0] == img_height, f"Image file {img.shape} not of size {img_height}, {img_width}"
    # global white_key_width_tolerence, black_key_width_tolerence
    white_imgs = []
    black_imgs = []

    wt = white_key_width_tolerence if bundle else white_key_width_strict
    bt = black_key_width_tolerence if bundle else black_key_width_strict

    for i in range(52):
        left = max(0, i * white_key_width_strict - wt)
        right = min(img_width, (i + 1) * white_key_width_strict + wt)
        up = 0
        down = img_height
        white_imgs.append((left, right, up, down))
    for i in [1, 4, 6, 9, 11, 13, 16, 18, 21, 23, 25, 28, 30, 33, 35, 37, 40, 42, 45, 47, 49, 52, 54, 57, 59, 61, 64,
              66, 69, 71, 73, 76, 78, 81, 83, 85]:
        left = max(0, i * black_key_width_strict - bt)
        right = min(img_width, (i + 1) * black_key_width_strict + bt)
        up = 0
        down = img_height
        black_imgs.append((left, right, up, down))
    return white_imgs, black_imgs


def separate(img, bundle=False):
    """
    img: An interable, each element is a image file of a keyboard. The
    images should be standardized, i.e. rectangular of size 106, 884

    Crop the images to seperate keys and list out in this order: white keys
    from lowest to highest, black keys from lowest to highest

    return: Two lists of image files, white and black correspondingly.
    """
    assert img.shape[1] == img_width and img.shape[0] == img_height, f"Image file {img.shape} not of size {img_height}, {img_width}"
    white_boxes, black_boxes = get_bounding_box(img, bundle)

    white_imgs = [
        img[box[2]:box[3], box[0]:box[1], :].copy()
        for box in white_boxes
    ]
    black_imgs = [
        img[box[2]:box[3], box[0]:box[1], :].copy()
        for box in black_boxes
    ]
    return white_imgs, black_imgs


if __name__ == "__main__":
    import cv2
    import os
    import glob
    import numpy as np
    import random
    import tqdm

    path = {
        'K_train': 'dataset/K_train',
        'K_test': 'dataset/K_test',
        'K_val': 'dataset/K_val'
    }

    X_path = dict()
    y = dict()


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
    for x in tqdm.tqdm(X):
        for path in tqdm.tqdm(X[x]):
            img = cv2.imread(path)
            img = np.transpose(img, (1, 0, 2))
            white, black = separate(img)
            white_num = 0
            black_num = 0
            new_path = path.replace('K', 'S')
            new_path = new_path.replace('.jpg', '_')
            for w in white:
                white_num += 1
                cv2.imwrite(f'{new_path}w_{white_num}.jpg', np.transpose(w, (1, 0, 2)))
            for b in black:
                black_num += 1
                cv2.imwrite(f'{new_path}b_{black_num}.jpg', np.transpose(b, (1, 0, 2)))
