# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 19:27:16 2022

@author: Sharad
"""
import cv2
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


def pool_image(img):
    
    f_pooling = 2
    step = 2

    height_out = int((img.shape[0] - 2)/2 +1)
    width_out = int((img.shape[1] - 2)/2 + 1)
    
    
    output_img = np.zeros((height_out, width_out))
    print(img.shape)
    print(output_img.shape)
    
    ii = 0;
    
    for i in range(0, img.shape[0] - f_pooling + 1, step):
        jj = 0;
        for j in range(0, img.shape[1] - f_pooling + 1, step):
            
            patch = img[i:i+f_pooling, j:j+f_pooling]
            
            output_img[ii, jj] = np.max(patch)
            
            jj += 1
        ii += 1
    
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

def gray_img(img):
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    return img1