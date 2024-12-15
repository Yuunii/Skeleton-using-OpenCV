import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

cap = cv2.VideoCapture("Walk.mp4")

#data읽기
while True:
    ret, img = cap.read()
    
    img = cv2.resize(img,(600, 400))
    
    results = pose.process(img)
    print(results.pose_landmarks)
    
    mp_draw.draw_landmarks(img, results.pose_landmarks,mp_pose.POSE_CONNECTIONS)
    
    
    cv2.imshow("Pose Estimation", img)
    cv2.waitKey(2)