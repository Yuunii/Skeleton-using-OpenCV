#  person landmarks using mediapipe
if you check out skeleton.py you will see that the person landmarks are now person landmarks with the background removed.

# Multi person detection with Yolov3 and tracking landmarks with mediapipe
single person detection can only mediapipe but multipose can't detection only medipipe
so We used yolov3 to detect bounding boxes for people, and then proceeded with landmarks detection using mediapipe for the bounding boxes.
if you check out multiperson_landmarks.py you will see multi person landmarks tracking
