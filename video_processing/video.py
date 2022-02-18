# -*- coding: utf-8 -*-

import cv2
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
from timeit import default_timer as timer
from utils.util import sobel_filter, laplacian_filter, prewitt_filter


f1 = prewitt_filter()

cv2.namedWindow("current view", cv2.WINDOW_NORMAL)

cv2.namedWindow("contour", cv2.WINDOW_NORMAL)

cv2.namedWindow("Cut fragement", cv2.WINDOW_NORMAL)

cv2.namedWindow("time Spent", cv2.WINDOW_NORMAL)

camera = cv2.VideoCapture(0)

h, w = None, None

temp = np.zeros((720, 1280, 3), np.uint8)

counter = 0

fps_start = timer()

while True:
    _, frame_bgr = camera.read()
    
    if not _:
        break
    
    if h is None or w is None:
        (h, w) = frame_bgr.shape[:2]
        
        layer = tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), strides=1, padding='same', activation='relu',
                                       input_shape=(h,w, 1),use_bias=False, kernel_initializer=tf.keras.initializers.constant(f1))
        
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    x_input_gray = frame_gray.reshape(1, h, w, 1).astype(np.float32)
    
    start = timer()
    
    output = layer(x_input_gray)

    end = timer()
    
    output = np.array(output[0, :, :, 0])
   
    
    output = np.clip(output, 0, 255).astype(np.uint8)
    
    #dilated = cv2.dilate(output, None, iterations=3)
    
    dilated = output
    
    v = cv2.__version__.split(".")[0]
    
    if v == '3':
        _, contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
    else:
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        
    contours = sorted(contours, key=cv2.contourArea, reverse = True)
    
    if contours:
        
        (x_min, y_min, box_width, box_height) = cv2.boundingRect(contours[0])
        
        cv2.rectangle(frame_bgr, (x_min, y_min), (x_min+box_width, y_min+box_height),
                      (0, 255, 0), 3)
        
        label = "fish"
        
        cv2.putText(frame_bgr, label, (x_min - 5, y_min - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        
        
        cut_fragment_bgr = frame_bgr[y_min + int(box_height * 0.1): y_min + box_height + int(box_height*0.1),
                                     x_min + int(box_width * 0.1): x_min + box_width + int(box_width)]
        
        cv2.imshow("current view", frame_bgr)
        cv2.imshow("contour", output)
        cv2.imshow("Cut fragement", cut_fragment_bgr)
        
        temp[:,:,0] = 230
        temp[:,:,1] = 161
        temp[:,:,2] = 0
        
        cv2.putText(temp, label, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 255, 255), 4, cv2.LINE_AA)
        
        cv2.putText(temp, f"time = {end - start}", (100, 600), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 4, cv2.LINE_AA)
        
        cv2.imshow("time Spent", temp)
        
    else:
        cv2.imshow("current view", frame_bgr)
        
        temp[:,:,0] = 230
        temp[:,:,1] = 161
        temp[:,:,2] = 0
        
        cv2.putText(temp, "no contour", (100, 450), cv2.FONT_HERSHEY_DUPLEX,  4, cv2.LINE_AA)
        cv2.imshow("contour", temp)
        cv2.imshow("Cut fragement", temp)
        cv2.imshow("time Spent", temp)
        
    counter +=1
        
    fps_stop = timer()
    
    if fps_stop - fps_start >= 1.0:
        print("fps: ", counter)
        counter = 0
        fps_start = timer()
        
    if cv2.waitKey(1) & 0xff == ord('q'):
        
        
        camera.release()
    
        cv2.destroyAllWindows()
        break
        