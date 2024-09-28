# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 22:37:30 2024

@author: cakir
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def matching(img1, img2):
    counter = 0

    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            counter += 1
            matchesMask[i]=[1,0]
            
    print(counter)
    if counter > 180:
        frame = img1
        return frame
    
    else:
        return None