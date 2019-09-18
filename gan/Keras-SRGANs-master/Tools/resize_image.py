import cv2
import os.path
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def load_path(path):
    directories = []
    if os.path.isdir(path):
        directories.append(path)
    for elem in os.listdir(path):
        if os.path.isdir(os.path.join(path, elem)):
            directories = directories + load_path(os.path.join(path, elem))
            directories.append(os.path.join(path, elem))
    return directories


def resize_image(img_src, dimension=(96, 96)):
    img = cv2.imread(img_src, cv2.IMREAD_UNCHANGED)
    resized = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
    print(resized.shape)
