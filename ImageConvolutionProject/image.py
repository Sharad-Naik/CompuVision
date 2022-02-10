# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.util import show_image_matplotlib

img1 = cv2.imread(r"ImageConvolutionProject/images/color.jpg")
img2 = cv2.imread(r"ImageConvolutionProject/images/blk_1.jpg")
img3 = cv2.imread(r"ImageConvolutionProject/images/blk_2.jpg")

img4 = cv2.imread(r"ImageConvolutionProject/images/color.jpg")
img5 = cv2.imread(r"ImageConvolutionProject/images/blk_1.jpg")
img6 = cv2.imread(r"ImageConvolutionProject/images/blk_2.jpg")


show_image_matplotlib([img1, img2, img3], 3)
show_image_matplotlib([img4, img5, img6], 2)

