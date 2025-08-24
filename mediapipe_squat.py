import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle_3d(a, b, c):
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # 범위 제한
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

cap = cv2.VideoCapture('squat_3.mp4')
if not cap.isOpened():
    print("비디오 열기 실패")
    exit()

count = 0
stage = None  # 'down' or 'up'

with mp_pose.Pose(model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]

            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

            left_angle = calculate_angle_3d(left_hip, left_knee, left_ankle)
            right_angle = calculate_angle_3d(right_hip, right_knee, right_ankle)
            angle = (left_angle + right_angle) / 2

            cv2.putText(image, f"Angle: {int(angle)}", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 상태 판별 및 카운트
            if angle < 90:
                stage = 'down'
            if angle > 120 and stage == 'down':
                stage = 'up'
                count += 1
                print(f"스쿼트 횟수: {count}")

            cv2.putText(image, f"Squat Count: {count}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

        cv2.imshow('Squat Counter', image)

        key = cv2.waitKey(10) & 0xFF
        if key == 27:  # ESC 누르면 종료
            print("사용자 종료")
            break

cap.release()
cv2.destroyAllWindows()