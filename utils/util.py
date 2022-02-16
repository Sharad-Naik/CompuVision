# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 19:27:16 2022

@author: Sharad
"""

import numpy as np
from tqdm import tqdm

def sobel_filter():
    filter1 = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]])
    
    return filter1


def laplacian_filter():
    filter1 = np.array([[0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]])
    
    return filter1


def prewitt_filter():
    filter1 = np.array([[1, 0, -1],
                    [1, 0, -1],
                    [1, 0, -1]])    
    
    return filter1

def sobel_h_filter():
    filter1 = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -1, -1]])    
    
    return filter1

def prewitt_h_filter():
    filter1 = np.array([[1, 1, 1],
                    [0, 0, 0],
                    [-1, -1, -1]])    
    
    return filter1

def gausian_filter():
    filter1 = (1 / 16) * np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]])    
    
    return filter1

def conv_image(img, filters):
    l = len(filters)
    
    """
    preparing hyperparameters for convolution
    filter size = 3
    stride (step) = 1
    pad = 1
    
    height_out = (height_in - f_size + 2 * pad)/step + 1
    
    """
    img = np.pad(img, (1, 1), mode='constant', constant_values=0)
    
    output_img = np.zeros(tuple([l])+img.shape)

    print(output_img.shape)
    
    
    """
    convolution starts here
    """   
    for i in tqdm(range((img.shape[0])-2)):
        for j in range((img.shape[1]-2)):
            patch = img[i:i+3, j:j+3]
            
            for k in range(l):
                output_img[k, i, j] = np.sum(patch*filters[k])

            
    """
    end of convolution
    """
    
    #exclude values that are less than 0
    
    output_img = np.clip(output_img, 0, 255)
    
    return output_img


def image_filter(img, filters):
    #print("img",img.shape)
    img = np.pad(img, (1, 1), mode='constant', constant_values=0)
    
    #print("pad img",img.shape)
    output_img = np.zeros(img.shape)
    
    
    for i in tqdm(range((img.shape[0])-2)):
        for j in range((img.shape[1]-2)):
            patch = img[i:i+3, j:j+3]
            
            output_img[i, j] = np.sum(patch*filters)
            
    output_img = np.clip(output_img, 0, 255)
            
    return output_img