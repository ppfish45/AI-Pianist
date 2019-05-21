import os
import os.path as path
for cdir, dirs, files in os.walk('.'):
    for f in files:
        spl = path.splitext(f)
        if spl[1] == '.mid':
            subprocess.run(['timidity', path.abspath(os.path.join(cdir, f)), '-Ow', '-o', path.abspath(path.join(cdir, spl[0] + '.wav'))])
