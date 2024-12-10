# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 21:04:21 2024
Skeleton_2

@author: Jinwoo
"""

#먼저 아나콘다 아나콘다 프롬프트 창을 실행시키고,
# pip install mediapipe 를 기입하고 인스톨시킨다.
# 인스톨이 끝나면 아래 소스코드 15행과 같이 
# import mediapipe as mp 를 기입한다.

import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

pose = mp_pose.Pose()

cap = cv2.VideoCapture("Walk.mp4")

#data읽기
while True:
    ret, img = cap.read()
    
    img = cv2.resize(img,(600, 400))
    
    results = pose.process(img)
    
    #형태가 카메라에 들어오면 오른쪽 하단의 Console창에 정보를 보여준다
    print(results.pose_landmarks)    
    
    cv2.imshow("Pose Estimation", img)
    cv2.waitKey(2)