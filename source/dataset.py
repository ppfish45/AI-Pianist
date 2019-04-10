import os
import re
import cv2

PATH = os.path.join('..', 'Dataset')
TRAIN_PATH = os.path.join(PATH, 'TrainSet')
TEST_PATH = os.path.join(PATH, 'TestSet')

print("Load all file locations ...")

train_walk = os.walk(TRAIN_PATH)
test_walk = os.walk(TEST_PATH)

train_videos = []
test_videos = []
train_midi = []
test_midi = []

for root_path, dir_list, file_list in train_walk:
  for file_name in file_list:
    path = os.path.join(root_path, file_name)
    ext_name = path[-3 : ]
    if ext_name in ['wmv', 'avi', 'mpg', 'mp4']:
      train_videos.append(path)
    if ext_name == 'mid':
      train_midi.append(path)

for root_path, dir_list, file_list in test_walk:
  for file_name in file_list:
    path = os.path.join(root_path, file_name)
    ext_name = path[-3 : ]
    if ext_name in ['wmv', 'avi', 'mpg', 'mp4']:
      test_videos.append(path)
    if ext_name == 'mid':
      test_midi.append(path)

train_videos.sort()
#train_midi.sort()
test_videos.sort()
#test_midi.sort()

print('Training Dataset: %d videos.' % len(train_videos))
print('Testing Dataset: %d videos.' % len(test_videos))

class video:
  def __init__(self, index, is_train_data=True):
    if is_train_data:
      self.cap = cv2.VideoCapture(train_videos[index])
    else:
      self.cap = cv2.VideoCapture(test_videos[index])

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

def get_video_num(is_train_data=True):
  if is_train_data:
    return len(train_videos)
  else:
    return len(test_videos)
