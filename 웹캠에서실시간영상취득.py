"""
2024.12.02
웹캠에서 실시간 영상취득
작성자 : 김진우

"""

import cv2 
 
vid_capture = cv2.VideoCapture(0)

while(vid_capture.isOpened()):

  ret, frame = vid_capture.read()
  if ret == True:
    cv2.imshow('Frame',frame)
    
    key = cv2.waitKey(20)
     
    if key == ord('q'):
      break
  else:
    break
 
vid_capture.release()
cv2.destroyAllWindows()