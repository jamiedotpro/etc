import os
from shutil import copy, copyfile
import sys
import pathlib

ROOT_DIR = sys.argv[1]

os.chdir(ROOT_DIR)
print('>>>>>' + os.path.dirname(os.path.realpath(__file__)))

list = os.listdir('.')

top_files = []
for i in range(0, len(list)):
    # change to child directories
    os.chdir(list[i])
    print('>>>>>' + os.path.dirname(os.path.realpath(__file__)))

    # list files in child directory
    files = os.listdir('.')

    # sorting file with file size
    files_size = [*map(lambda file: os.stat(file).st_size, files)]
    files_size, files = zip(*sorted(zip(files_size, files)))

    file_path = ROOT_DIR + '/' + list[i]

    new_files = [*map(lambda file: file_path + '/' + file, files[-3:])]
    print(new_files)
    top_files += new_files

    os.chdir('../')

os.chdir('../')
print('>>>>>' + os.path.dirname(os.path.realpath(__file__)))

# move file to correct folder
OUT_DIR = 'VN_dataset'
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

for i in range(len(top_files)):
    copyfile(top_files[i], OUT_DIR + '/' + str(i) + ''.join(pathlib.Path(top_files[i]).suffixes))

print('>>>> DONE')
