# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 21:00:45 2022

@author: Shruti
"""


import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.util import prewitt_filter, gray_img


def convolution_2d(feature_map, f, pad = 1, step = 1):
    """
    f_size = filter size
    step (stride)
    pad
    
    height_out = (height_in - f_size + 2 * pad)/step + 1
    width_out = (width_in - fsize + 2 * pad)/step + 1
    """

    f_size = f.shape[0]
    height_out = int((feature_map.shape[0] - f_size + 2* pad)/step + 1)
    width_out = int((feature_map.shape[1] - f_size + 2* pad)/step + 1)
    
    feature_map_pad = np.pad(feature_map, (pad, pad), mode='constant',
                             constant_values=0)
    
    output_feature_map = np.zeros((height_out, width_out))
    
    for i in tqdm(range(feature_map.shape[0] - f_size + 1)):
        for j in range((feature_map.shape[1] - f_size + 1)):
            
            patch = feature_map[i: i+f_size, j: j+f_size]
            
            output_feature_map[i, j] = np.sum(patch*f)
            
    return output_feature_map


def range_0_255(feature_map):
    
    feature_map =np.clip(feature_map, 0, 255)
    
    return feature_map


def pooling(feature_map, f_pooling=2, step=2):
    
    height_out = int((feature_map.shape[0] - 2)/2 + 1)
    width_out = int((feature_map.shape[1]-2)/2 + 1)
    
    
    output_feature_map = np.zeros((height_out, width_out))
    
    ii = 0
    
    for i in range(0, feature_map.shape[0] - f_pooling + 1, step):
        jj = 0
        for j in range(0, feature_map.shape[1] - f_pooling + 1, step):
            
            patch = feature_map[i:i+f_pooling, j: j+f_pooling]
            
            output_feature_map[ii, jj] = np.max(patch)
            
            jj += 1
        ii += 1
            
    return output_feature_map
    


f1 = sobel_filter()
f2 = prewitt_filter()

img = cv2.imread(r"images/avenger.jpg")

################
# convolution
################

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

feature_map = gray_img(img)

output_feature_map = convolution_2d(feature_map, f2, pad = 1, step = 1)

output_feature_map = range_0_255(output_feature_map)

output_feature_map = pooling(output_feature_map)

output_feature_map = convolution_2d(output_feature_map, f2, pad = 1, step = 1)

output_feature_map = range_0_255(output_feature_map)

output_feature_map = pooling(output_feature_map)

output_feature_map = convolution_2d(output_feature_map, f2, pad = 1, step = 1)

output_feature_map = range_0_255(output_feature_map)

output_feature_map = pooling(output_feature_map)

plt.imshow(output_feature_map, cmap=plt.get_cmap('gray'))


