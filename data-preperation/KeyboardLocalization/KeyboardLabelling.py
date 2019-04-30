import cv2
import matplotlib.pyplot as plt
import skimage
import numpy as np
import imageio
import copy
import os
import sys
import glob

sys.path.append('..')

import my_utils
import dataset

cur_point = (0, 0)
lst_image = None
ref_image = None

dataset.init(os.path.join('..', '..', 'DataSet', 'Youtube'))

def click_and_choose(event, x, y, flags, param):
  global ref_image, lst_image, cur_point
  if event == cv2.EVENT_LBUTTONDOWN:
    cur_point = (int(x), int(y))
    ref_image = lst_image.copy()
    cv2.circle(ref_image, cur_point, 2, (0, 0, 255), 2)

def label(src_image):
  global ref_image, lst_image, cur_point
  text = ['top left', 'top right', 'bottom right', 'bottom left']
  pts = np.zeros([4, 2], dtype=np.float32)
  ref_image = src_image.copy()
  lst_image = src_image.copy()
  cur_stage = 0
  
  cv2.namedWindow('Labelling Window')
  cv2.setMouseCallback("Labelling Window", click_and_choose)

  while True:

    tmp_image = ref_image.copy()
    if cur_stage < 4:
      cv2.putText(tmp_image,
                  "Please choose the %s point on the keyboard and press 'c' to confirm" % text[cur_stage], (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                  0.4, (0, 0, 255), 1)
    cv2.imshow("Labelling Window", tmp_image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
      if cur_stage == 4:
        break
      else:
        pts[cur_stage] = [*cur_point]
        lst_image = ref_image.copy()
      cur_stage += 1
      if cur_stage == 4:
        # standard piano keyboard size
        width = 875
        height = 105
        dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts, dst)
        transformed_image = cv2.warpPerspective(src_image, M, (width, height))
        cv2.imshow('Cropped Keyboard', transformed_image)
        # imageio.imwrite('label_window.png', cv2.cvtColor(tmp_image, cv2.COLOR_BGR2RGB))
        # imageio.imwrite('cropped.png', cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
    elif key == ord('q'):
      break

  return pts

def label_video():

  print('Please input the starting index : ', end='')
  index = int(input())

  print('Please input the starting index of file name : ', end='')
  offset = int(input())

  os.makedirs(os.path.join('y_test'), exist_ok=True)
  for i in range(index, dataset.get_video_num()):
    X = []
    y = []
    print('Label video #%d ...' % (i + 1))
    v = dataset.video(i)
    n_frame = v.get_num_of_frames()
    sample = v.get_frame(250)
    pts = label(sample)
    os.makedirs(os.path.join('X_test', str(i - index + offset)), exist_ok=True)
    counter = 0
    for j in range(250, n_frame - 450, (n_frame - 500) // 200):
      frame = cv2.cvtColor(v.get_frame(j), cv2.COLOR_BGR2RGB)
      imageio.imwrite(os.path.join('X_test', str(i - index + offset), str(counter) + '.jpg'), frame)
      counter += 1
      y.append(np.array(pts))
    y = np.array(y)
    np.save(os.path.join('y_test', str(i - index + offset)), y)
    
def label_photo_dir(path):
  files = glob.glob(path + '/*.jpg')
  print(files)
  sample = cv2.imread(files[0])
  pts = label(sample)
  y = [pts] * len(files)
  np.save('y', y)

path = '../../models/keyboard-detection/dataset'
label_photo_dir(path + '/X_test/6')