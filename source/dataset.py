import os
import re

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
    if path[-3 : ] == 'wmv':
      train_videos.append(path)
    if path[-3 : ] == 'mid':
      train_midi.append(path)

for root_path, dir_list, file_list in test_walk:
  for file_name in file_list:
    path = os.path.join(root_path, file_name)
    if path[-3 : ] == 'wmv':
      test_videos.append(path)
    if path[-3 : ] == 'mid':
      test_midi.append(path)

train_videos.sort()
train_midi.sort()
test_videos.sort()
test_midi.sort()

print('Training Dataset: %d videos.' % len(train_videos))
print('Testing Dataset: %d videos.' % len(test_videos))