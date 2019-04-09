import os
import os.path as path
import re

pattern = re.compile(r'^\d+:\d+:\d+:\d+$')

okay = []
trailing = []
empty = []
corrupt = []

for cdir, dirs, files in os.walk('.'):
    for f in files:
        if f.endswith('.wmv.txt'):
            filepath = path.join(cdir, f)
            print('Checking file {}...'.format(filepath))
            with open(filepath) as fp:
                lines = fp.readlines()
            if len(lines) == 0:
                empty.append(filepath)
            elif len(lines) == 2 or (len(lines) == 3 and lines[2] == ''):
                passed = (lines[0] == '30') and bool(pattern.match(lines[1]))
                if len(lines) == 3 and lines[2] == '':
                    trailing.append(filepath)
                else:
                    okay.append(filepath)
            else:
                corrupt.append(filepath)


output = """Label format sanity Check:
==========
Passed:

{}
==========
Has trailing newline:

{}
==========
Is empty:

{}
==========
Is corrupted:

{}
""".format('\n'.join(okay), '\n'.join(trailing), '\n'.join(empty), '\n'.join(corrupt))


with open('./label_sanity_check_result.txt', 'w') as fp:
    fp.write(output)


print("==============")
print('Please check the result in "label_sanity_check_result.txt"')