import cv2
import dlib
from detect import Detection
from webcam import Realtime


def main():
    # Haar Cascade 및 Dlib 모델 초기화
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # 랜드마크 포인트
    left_eye_points = list(range(36, 42))  # 왼쪽 눈 포인트
    right_eye_points = list(range(42, 48))  # 오른쪽 눈 포인트

    # Detection 클래스 초기화
    detection = Detection(face_cascade, predictor, right_eye_points, left_eye_points)

    # Realtime 얼굴 탐지 실행
    realtime = Realtime(detection)
    realtime.start()


if __name__ == "__main__":
    main()