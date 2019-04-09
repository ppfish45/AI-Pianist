import os
import os.path as path

for cdir, dirs, files in os.walk('.'):
    for f in files:
        spl = path.splitext(f)
        if spl[1] == '.wmv':
            with open(path.join(cdir, f+'.txt'), 'w'):
                pass

