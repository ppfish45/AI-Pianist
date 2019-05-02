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

assert black_key_height == 69 and black_key_width == 10 and white_key_height == 106 and white_key_width == 17, "Incorrect calculation of key dimentions"

def seperate(img):
    """
    img: An interable, each element is a image file of a keyboard. The
    images should be standardized, i.e. rectangular of size 884, 106

    Crop the images to seperate keys and list out in this order: white keys
    from lowest to highest, black keys from lowest to highest

    return: A list of image files .
    """
    assert img.shape[0] == img_height and img.shape[1] == img_height, f"Image file not of size {img_width}, {img_height}"
    white_imgs = [
        img[max(0, i*white_key_width_strict-white_key_width_tolerence):min(img_width, (i+1)*white_key_width_strict+white_key_width_tolerence), :].copy() 
        for i in range(52)
    ]
    black_imgs = [
        img[max(0, i*black_key_width_strict-black_key_width_tolerence):min(img_width, (i+1)*black_key_width_strict+black_key_width_tolerence), :].copy() 
        for i in [1, 4, 6, 9, 11, 13, 16, 18, 21, 23, 25, 28, 30, 33, 35, 37, 40, 42, 45, 47, 49, 52, 54, 57, 59, 61, 64, 66, 69, 71, 73, 76, 78, 81, 83, 85]
    ]
    return white_imgs + black_imgs
    