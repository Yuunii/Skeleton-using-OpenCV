import numpy as np
import cv2
import dlib


class Detection():
    def __init__(self, face_cascade, predictor, Right_eye_point, Left_eye_point):
        self.face_cascade = face_cascade
        self.predictor = predictor
        self.Left_eye_point = Left_eye_point
        self.Right_eye_point = Right_eye_point

    def detect(self, gray, frame):
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(100,100), flags= cv2.CASCADE_SCALE_IMAGE)

        for (x,y,w,h) in faces:
            dlib_rect = dlib.rectangle(int(x),int(y), int(x+w),int(y+h))
            landmarks = np.matrix([[p.x,p.y] for p in self.predictor(frame, dlib_rect).part()])
            landmarks_display = landmarks[self.Right_eye_point, self.Left_eye_point]

        for idx, point in enumerate(landmarks_display):
            pos = (point[0,0],point[0,1])
            cv2.circle(frame, pos, 2, color=(0,255,255), thickness=-1)

        return frame


