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