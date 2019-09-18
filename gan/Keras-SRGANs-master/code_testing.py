# Code testing: where to learn python and test code
import os, os.path
from random import sample

# [File counting]
# for root, dirs, files in os.walk("./data", topdown=False):
#     print(len(files))

# [Delete files]
# files = os.listdir('./data')
# # output: List
# for file in sample(files, 2):
#     os.remove('./data/' + file)

# # [Get size(byte) image]
# files = os.listdir('./data/')
# # Sort image by size
# files_sorted = sorted(files, key=lambda file_: os.path.getsize('./data_hr/' + file_), reverse=True)

#

# # Move 3 largest size to other folder
# for file in files_sorted:
#     print(os.path.getsize('./data_hr/' + file))

# dir = os.fsencode('VN-celeb')
# for d in dir:
#     os.chdir(d)
#
#     dir2 = os.fsencode()

# list = os.listdir('./VN_dataset/')
#
# print(len(list))

a = [2, 1, 3, 5, 3, 2]

def firstDuplicate(a):
    duplicate_list = []
    for i in range(len(a) - 1):
        for j in range(i + 1, len(a)):
            if a[i] == a[j]:
                duplicate_list.append(j)
    if len(duplicate_list) == 0:
        return -1
    return a[min(duplicate_list)]

firstDuplicate(a)