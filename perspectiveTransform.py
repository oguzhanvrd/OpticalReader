# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 23:59:47 2024

@author: cakir
"""

import cv2
import numpy as np
from detectCorners import detectCorners
import time

def applyPerspectiveTransform(image):
    image = cv2.resize(image, (450, 600))

    corners = detectCorners(image)

    if len(corners) == 6:
        tl = corners[4][0]
    
    print('bulunan kenarlar: ', len(corners))
    tl = corners[0][0]
    bl = corners[1][0]
    br = corners[2][0]
    tr = corners[3][0]
    
    pts1 = np.float32([tl, tr, bl, br])
    pts2 = np.float32([[0,0], [480, 0], [0, 640], [480, 640]])
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    
    transformed_frame = cv2.warpPerspective(image, matrix, (480, 640))
    

    return transformed_frame