import cv2
import mediapipe as mp
import numpy as np
import time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7,
                    min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

FRONT_THRESHOLD = 0.25
SLIGHT_TURN_THRESHOLD = 0.18

TILT_TOLERANCE = 0.03
SHOULDER_TOLERANCE = 0.03
NECK_TOLERANCE = 0.03

bad_start_time = None

def calculate_vertical_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = abs(np.degrees(np.arctan2(dy, dx)))
    return angle

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
        right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]

        shoulder_distance = abs(left_shoulder.x - right_shoulder.x)

        shoulder_center = [(left_shoulder.x + right_shoulder.x) / 2,
                           (left_shoulder.y + right_shoulder.y) / 2]

        hip_center = [(left_hip.x + right_hip.x) / 2,
                      (left_hip.y + right_hip.y) / 2]

        ear_center = [(left_ear.x + right_ear.x) / 2,
                      (left_ear.y + right_ear.y) / 2]

        body_tilt = abs(shoulder_center[0] - hip_center[0])
        shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
        neck_shift = abs(ear_center[0] - shoulder_center[0])

        spine_angle = calculate_vertical_angle(hip_center, shoulder_center)

        # ---------------- POSTURE SCORE ----------------
        posture_score = 100

        if body_tilt > TILT_TOLERANCE:
            posture_score -= 25
        if shoulder_diff > SHOULDER_TOLERANCE:
            posture_score -= 25
        if neck_shift > NECK_TOLERANCE:
            posture_score -= 25
        if shoulder_distance < SLIGHT_TURN_THRESHOLD:
            posture_score -= 25

        # ---------------- STATUS ----------------
        if posture_score >= 75:
            status = "GOOD POSTURE"
            color = (0, 255, 0)
            bad_start_time = None
        elif 50 <= posture_score < 75:
            status = "PLEASE SIT STRAIGHT"
            color = (0, 255, 255)
            bad_start_time = None
        else:
            status = "BAD POSTURE"
            color = (0, 0, 255)

            if bad_start_time is None:
                bad_start_time = time.time()

        # ---------------- DISPLAY ----------------
        cv2.putText(image, status, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        cv2.putText(image, f"Score: {posture_score}%",
                    (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

        cv2.putText(image, f"Spine Angle: {int(spine_angle)} deg",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        if bad_start_time:
            duration = int(time.time() - bad_start_time)
            cv2.putText(image, f"Bad Posture Time: {duration}s",
                        (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)

        mp_drawing.draw_landmarks(image,
                                  results.pose_landmarks,
                                  mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Smart Posture Intelligence System", image)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()