import numpy as np
import cv2

img_width = 884
img_height = 106

white_key_count = 52
black_key_count = 36

white_key_width_strict = 17 # 884 / 52
white_key_width_tolerence = 2 # total 21
black_key_width_strict = 12 # observed value
black_key_width_tolerence = 2 # observed value
ocvate_width_strict = white_key_width_strict * 7
white_key_height = img_height
black_key_height = img_height

black_spacing = 8 # total spacing is wider than 8*7 by 3

white_key_width = white_key_width_tolerence * 2 + white_key_width_strict  # 21
white_key_width_bundle = 3 * white_key_width_strict  # 51

black_key_width = black_key_width_tolerence * 2 + black_key_width_strict  # 16
black_key_width_bundle = 3 * black_key_width_strict  # 24

assert black_key_height == 106 and black_key_width == 16 and white_key_height == 106 and white_key_width == 21, "Incorrect calculation of key dimentions "

def record_black_here(left, tleft, tright):
    return (left - tleft, left + black_key_width_strict + tright, 0, black_key_height)


def get_bounding_box(img, bundle=False):
    """
    img: An image file of a keyboard. The
    images should be standardized, i.e. rectangular of size 106, 884

    Get cropping bounding boxes of seperate keys and list out in this order: white keys
    from lowest to highest, black keys from lowest to highest

    return: A list of dimension 4 bounding boxes: left, right, up, down. Can be out of bound.
    """
    assert img.shape[1] == img_width and img.shape[0] == img_height, f"Image file {img.shape} not of size {img_height}, {img_width}"

    wt = white_key_width_strict if bundle else white_key_width_tolerence
    bt = black_key_width_strict if bundle else black_key_width_tolerence

    white_imgs = [(i * white_key_width_strict - wt, (i + 1) * white_key_width_strict + wt, 0, img_height) for i in range(52)]
    black_imgs = []

    black_imgs.append(record_black_here(11, bt, bt)) # hardcode the position of the first black key
    octave_starts = [i * white_key_width_strict for i in (2, 9, 16, 23, 30, 37, 44)]
    for octave_count, cur in enumerate(octave_starts):
        if octave_count == 0:
            starting_spacing = black_spacing
        elif octave_count == 1:
            starting_spacing = black_spacing + 1
        else:
            starting_spacing = black_spacing + 2
        if octave_count < 5:
            middle_spacing = black_spacing * 2 + 2
        else:
            middle_spacing = black_spacing * 2 + 3
        cur += starting_spacing
        black_imgs.append(record_black_here(cur, bt, bt))
        cur += black_key_width_strict + black_spacing
        black_imgs.append(record_black_here(cur, bt, bt))
        cur += black_key_width_strict + middle_spacing
        black_imgs.append(record_black_here(cur, bt, bt))
        cur += black_key_width_strict + black_spacing
        black_imgs.append(record_black_here(cur, bt, bt))
        cur += black_key_width_strict + black_spacing
        if octave_count < 4 or octave_count == 6 or bundle:
            black_imgs.append(record_black_here(cur, bt, bt))
        else:
            black_imgs.append(record_black_here(cur, bt*2, 0))

    return white_imgs, black_imgs


def get_black_keys(img, expected_width=20):
    '''
    want a RGB image, (h, w) format
    return: a list of a tuple of (left, right, top, bottom) coordinates.
    '''
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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
                black_keys.append([last, i - 1, 0, black_key_height])
                last = -1

    for i, coor in enumerate(black_keys):
        x, y, _, _ = coor
        offset = (expected_width - (y - x)) // 2
        if y - x + offset * 2 == expected_width:
            coor[0] = x - offset
            coor[1] = y + offset
        else: # prevent //2 rounding from decreasing the size by 1
            if i < 18:
                coor[0] = x - offset - 1
                coor[1] = y + offset
            else:
                coor[0] = x - offset
                coor[1] = y + offset + 1
    # print(len(black_keys))
    return black_keys


def get_bounding_box2(img, bundle=False):
    """
    img: An image file of a keyboard. The
    images should be standardized, i.e. rectangular of size 106, 884

    Get cropping bounding boxes of seperate keys and list out in this order: white keys
    from lowest to highest, black keys from lowest to highest

    return: A list of dimension 4 bounding boxes: left, right, up, down. Can be out of bound (etc. negative for first white key).
    """
    assert img.shape[1] == img_width and img.shape[0] == img_height, f"Image file {img.shape} not of size {img_height}, {img_width}"

    wt = white_key_width_strict if bundle else white_key_width_tolerence
    bt = black_key_width_strict if bundle else black_key_width_tolerence

    white_imgs = [(i * white_key_width_strict - wt, (i + 1) * white_key_width_strict + wt, 0, img_height) for i in range(52)]
    black_imgs = get_black_keys(img, expected_width=black_key_width)
    # Always get bounding box of width black_strict+tolerence. Add bundle later to avoid interference in cv
    if bundle:
        diff = black_key_width_strict - black_key_width_tolerence
        for coor in black_imgs:
            coor[0] -= diff
            coor[1] += diff
    return white_imgs, black_imgs


def separate(img, bundle=False):
    """
    img: An image file of a keyboard. The
    images should be standardized, i.e. rectangular of size 106, 884

    Crop the images to seperate keys and list out in this order: white keys
    from lowest to highest, black keys from lowest to highest

    return: Two lists of image files, white and black correspondingly.
    """
    assert img.shape[1] == img_width and img.shape[
        0] == img_height, f"Image file {img.shape} not of size {img_height}, {img_width}"
    white_boxes, black_boxes = get_bounding_box(img, bundle)

    white_imgs = [
        img[box[2]:box[3], max(0, box[0]):min(img_width, box[1]), :].copy()
        for box in white_boxes
    ]
    black_imgs = [
        img[box[2]:box[3], max(0, box[0]):min(img_width, box[1]), :].copy()
        for box in black_boxes
    ]
    for img_array in (white_imgs, black_imgs):
        if img_array[0].shape[1] < img_array[1].shape[1]:
            pad_width = img_array[1].shape[1] - img_array[0].shape[1]
            img_array[0] = np.pad(img_array[0], ((0, 0), (pad_width, 0), (0, 0)), mode='constant', constant_values=0)
        if img_array[-1].shape[1] < img_array[-2].shape[1]:
            pad_width = img_array[-2].shape[1] - img_array[-1].shape[1]
            img_array[-1] = np.pad(img_array[-1], ((0, 0), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
    return white_imgs, black_imgs


if __name__ == "__main__":
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
