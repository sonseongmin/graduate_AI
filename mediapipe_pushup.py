import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 3D 각도 계산 함수
def calculate_angle_3d(a, b, c):
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# 팔꿈치 각도 평균 계산
def calculate_pushup_angle(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

    left_angle = calculate_angle_3d(left_shoulder, left_elbow, left_wrist)
    right_angle = calculate_angle_3d(right_shoulder, right_elbow, right_wrist)

    return left_angle, right_angle, (left_angle + right_angle) / 2

# 비디오 로드
cap = cv2.VideoCapture('push_up_2.mp4')
if not cap.isOpened():
    print("비디오 열기 실패")
    exit()

count = 0
stage = None  # 'down' 또는 'up'

# Pose 모델 실행
with mp_pose.Pose(model_complexity=2,
                  min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 전처리
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 포즈 랜드마크가 있을 경우
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # 팔꿈치 각도 계산 (평균)
            left_angle, right_angle, avg_angle = calculate_pushup_angle(landmarks)

            # 상태 판별 및 카운트
            if avg_angle < 100:
                stage = 'down'
            if avg_angle > 150 and stage == 'down':
                stage = 'up'
                count += 1
                print(f"푸시업 횟수: {count}")

            # 각도 및 카운트 표시
            cv2.putText(image, f"Avg Angle: {int(avg_angle)}", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.putText(image, f"Pushup Count: {count}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            # 랜드마크 시각화
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

        # 화면 출력용 축소
        display_image = cv2.resize(image, (640, 360))  # 화면 크기 조정
        cv2.imshow('Pushup Counter', display_image)

        # ESC 키로 종료
        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            print("사용자 종료")
            break

cap.release()
cv2.destroyAllWindows()
