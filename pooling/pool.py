import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from utils.util import gray_img, pool_image

from tqdm import tqdm

img = cv2.imread(r"images\beach.jpg")

img = gray_img(img)

plt.imshow(img, cmap=plt.get_cmap('gray'))



"""
height_out = (height_in - f_pooling)/step + 1
width_out = (width_in -f_pooling)/step + 1
"""
plt.show()

output_img = pool_image(img)

plt.imshow(output_img, cmap = plt.get_cmap("gray"))