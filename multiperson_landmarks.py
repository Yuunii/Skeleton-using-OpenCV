import cv2
import mediapipe as mp
import numpy as np

# YOLOv3 모델 설정
yolo_config = 'yolov3.cfg'
yolo_weights = 'yolov3.weights'
yolo_classes = 'coco.names'

net = cv2.dnn.readNetFromDarknet(yolo_config, yolo_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

with open(yolo_classes, 'r') as f:
    classes = f.read().strip().split('\n')

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Mediapipe 설정
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()


def detect_persons_yolo(frame):
    """YOLOv3를 이용하여 사람 감지"""
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "person":  # 사람만 감지
                box = detection[0:4] * np.array([width, height, width, height])
                center_x, center_y, w, h = box.astype('int')
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # NMS 적용
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.6)

    final_boxes = []
    if isinstance(indices, np.ndarray):  # indices가 ndarray인지 확인
        for i in indices.flatten():  # flatten()을 사용하여 1차원으로 변환
            final_boxes.append(boxes[i])
    elif isinstance(indices, list):  # 만약 리스트 형태로 반환된 경우
        final_boxes = [boxes[i[0]] for i in indices]

    return final_boxes


def main():
    cap = cv2.VideoCapture("motion.mp4")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv3로 사람 검출
        person_boxes = detect_persons_yolo(frame)

        # 각 바운딩 박스에 대해 Mediapipe Pose 적용 및 랜드마크 연결
        skeleton_image = np.zeros_like(frame)
        for box in person_boxes:
            x, y, w, h = box
            cropped_frame = frame[y:y + h, x:x + w]
            if cropped_frame.size == 0:
                continue
            cropped_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            results = pose.process(cropped_rgb)

            if results.pose_landmarks:
                # Mediapipe로 랜드마크와 연결선 그리기
                mp_draw.draw_landmarks(
                    skeleton_image[y:y + h, x:x + w],
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
                )

        # 스켈레톤 랜드마크를 작게 출력
        resized_skeleton = cv2.resize(skeleton_image, (640, 360))  # 영상 크기 조정
        cv2.imshow("Skeleton Landmarks", resized_skeleton)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()