# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.util import laplacian_filter, sobel_filter, prewitt_filter, \
    conv_image, gausian_filter, prewitt_h_filter, sobel_h_filter, image_filter


img1 = cv2.imread(r"ImageConvolutionProject/images/color.jpg")

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

img2 = cv2.imread(r"ImageConvolutionProject/images/blk_1.jpg")

img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

img3 = cv2.imread(r"ImageConvolutionProject/images/blk_2.jpg")

img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
img3 = cv2.cvtColor(img3, cv2.COLOR_RGB2GRAY)





"""
Filter for edge detection
"""
#sobel filter to detect vertical changes on image
filter1 = sobel_filter()

#laplacian filter for detecting different brightness on images
filter2 = laplacian_filter()

#prewitt filter to detect vertical changes on image
filter3 = prewitt_filter()

filter4 = gausian_filter()

filter5 = prewitt_h_filter()

filter6 = sobel_h_filter()

filters = []

"""
convolution starts here
"""   
output_img = conv_image(img1, filters)

print(img1.shape)
img2 = image_filter(img2, filter2)
img2 = image_filter(img2, filter4)
img2 = image_filter(img2, filter2)

img_length = len(output_img)


if img_length <=3:
    img_length = 4

plt.rcParams['figure.figsize'] = (10.0, 8.0)   

        
plt.imshow(img2, cmap=plt.get_cmap("gray"))

# figure, ax = plt.subplots(nrows=int(img_length/2), ncols=int(img_length/2))

# x = 0;
# for i in range(int(img_length/2)):
#     if x == len(filters):
#         break
#     for j in range(int(img_length/2)):
#         print(i, j)
#         ax[i, j].imshow(output_img[x], cmap=plt.get_cmap("gray"))
#         ax[i, j].set_title("Color")
#         x = x + 1;
       
# for i in range(int(img_length/2)):
#     for j in range(int(img_length/2)):
#         ax[i, j].axis("off")
        
# plt.imshow(oimg2, cmap=plt.get_cmap("gray"))

# plt.tight_layout()

# figure.canvas.set_window_title("Color and gray images")

plt.show()