import cv2
from detect import Detection

class Realtime:
    def __init__(self, detection):
        self.detection = detection

    def start(self):
        video_capture = cv2.VideoCapture(0)

        while True:
            _, frame = video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 얼굴 탐지 수행
            canvas = self.detection.detect(gray, frame)

            # 결과 화면 출력
            cv2.imshow('Realtime Face Detection', canvas)

            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()