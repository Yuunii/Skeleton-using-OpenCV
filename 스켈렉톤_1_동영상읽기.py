# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 20:57:55 2024

Skeleton_1

@author: Jinwoo
"""

import cv2


cap = cv2.VideoCapture("Walk.mp4")

#data읽기
while True:
    ret, img = cap.read()
    
    img = cv2.resize(img,(600, 400))          
    
    
    cv2.imshow("Pose Estimation", img)
    cv2.waitKey(1)