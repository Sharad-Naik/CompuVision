# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 19:27:16 2022

@author: Sharad
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

def show_image_matplotlib(imgs, shape_d):
    plt.rcParams['figure.figsize'] = (10.0, 10.0)

    if len(imgs) < 9:
        figure, ax = plt.subplots(nrows=3, ncols=3)
        cntr = 0;
        print(len(imgs))
        if shape_d == 3:
            for i in range(3):
                for j in range(3):
                    try:
                        ax[i, j].imshow(cv2.cvtColor(imgs[cntr], cv2.COLOR_BGR2RGB))
                        if cntr < len(imgs):
                                cntr = cntr+1
                    except:
                        pass
                            
        if shape_d == 2:
            for i in range(3):
                for j in range(3):
                    try:
                        ax[i, j].imshow(cv2.cvtColor(imgs[cntr], cv2.COLOR_BGR2GRAY))
                        if cntr < len(imgs):
                                cntr = cntr+1
                    except:
                        pass
