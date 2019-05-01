import os
import re
import cv2

PATH = os.path.join('..', 'Dataset', 'TrainSet')
videos = []

def init(path=None):
  global PATH
  if path:
    PATH = path

  print("Load all file locations ...")

  walk = os.walk(PATH)

  for root_path, dir_list, file_list in walk:
    for file_name in file_list:
      path = os.path.join(root_path, file_name)
      ext_name = path[-3 : ]
      if ext_name in ['wmv', 'avi', 'mpg', 'mp4']:
        videos.append(path)

  videos.sort()

  print('%d videos found.' % len(videos))

def get_video_num():
    return len(videos)

class video:
  def __init__(self, index):
    self.cap = cv2.VideoCapture(videos[index])

  def get_num_of_frames(self):
    return (int)(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

  def get_frame_size(self):
    return (self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT), self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

  def get_frame(self, index):
    self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    res, frame = self.cap.read()
    if res:
      return frame
    else:
      return None


