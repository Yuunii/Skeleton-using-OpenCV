import cv2
import mediapipe as mp
import numpy as np  # 추가로 numpy가 필요합니다.

def main():
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose()

    cap = cv2.VideoCapture("Walk.mp4")

    # 데이터 읽기
    while True:
        ret, img = cap.read()
        if not ret:
            break

        img = cv2.resize(img, (600, 400))

        # 스켈레톤을 그릴 빈 이미지 (검정 배경)
        skeleton_img = np.zeros_like(img)

        # Mediapipe로 포즈 추출
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            # 스켈레톤을 빈 이미지 위에 그리기
            mp_draw.draw_landmarks(
                skeleton_img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),  # 스켈레톤 선
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)       # 관절 포인트
            )

        # 결과 이미지 출력
        cv2.imshow("Skeleton Only", skeleton_img)

        if cv2.waitKey(2) & 0xFF == ord("q"):  # 'q' 키를 누르면 종료
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()