'''
This module is to rename the folders' names to a consecutive ascending order
'''

import os
import glob

path = 'y_train copy'

def rename_folder(path):
    folders = [x for x in os.listdir(path) if x[0] != '.']
    folders = sorted(folders, key=lambda x: int(x))
    for i, name in enumerate(folders):
        os.rename(os.path.join(path, name), os.path.join(path, str(i)))

def rename_files(path):
    folders = [x for x in os.listdir(path) if x[0] != '.']
    ext = '.' + folders[0].split('.')[1]
    folders = [x.split('.')[0] for x in folders]
    folders = sorted(folders, key=lambda x: int(x))
    for i, name in enumerate(folders):
        os.rename(os.path.join(path, name + ext), os.path.join(path, str(i) + ext))

rename_folder('X_train')
rename_folder('X_test')
rename_folder('X_val')

rename_files('y_train')
rename_files('y_test')
rename_files('y_val')