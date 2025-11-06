import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import cv2
import mediapipe as mp
import numpy as np

# --- 설정 ---
VIDEO_PATH = r"C:\Users\user\Downloads\KakaoTalk_20251014_224325480.mp4"  # 테스트할 영상 경로
EXERCISE = "pushup"

# 임계값 (푸쉬업 동작 범위)
LOW_TH, HIGH_TH = 125, 140
DEBOUNCE_FR = 3

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils       # 추가됨 ✅
mp_drawing_styles = mp.solutions.drawing_styles

# --- 각도 계산 함수 ---
def calculate_angle_3d(a, b, c):
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    ba, bc = a - b, c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom == 0:
        return 0.0
    cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))

# --- 양팔 평균 팔꿈치 각도 계산 ---
def avg_elbow_angle(lm):
    l = (lm[mp_pose.PoseLandmark.LEFT_SHOULDER],
         lm[mp_pose.PoseLandmark.LEFT_ELBOW],
         lm[mp_pose.PoseLandmark.LEFT_WRIST])
    r = (lm[mp_pose.PoseLandmark.RIGHT_SHOULDER],
         lm[mp_pose.PoseLandmark.RIGHT_ELBOW],
         lm[mp_pose.PoseLandmark.RIGHT_WRIST])
    return (calculate_angle_3d(*l) + calculate_angle_3d(*r)) / 2.0

# --- 비디오 처리 ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ 비디오를 열 수 없습니다:", VIDEO_PATH)
    exit()

count, stage = 0, None
hold_low = hold_high = 0

with mp_pose.Pose(model_complexity=2,
                  min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            lms = results.pose_landmarks.landmark
            angle = avg_elbow_angle(lms)

            # Down / Up 판별
            hold_low = hold_low + 1 if angle < LOW_TH else 0
            hold_high = hold_high + 1 if angle > HIGH_TH else 0

            if hold_low >= DEBOUNCE_FR:
                stage = "down"
            if stage == "down" and hold_high >= DEBOUNCE_FR:
                stage = "up"
                count += 1
                hold_low = hold_high = 0

            # ✅ MediaPipe 관절 그리기
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # 텍스트 표시
            cv2.putText(frame, f"{EXERCISE} Count: {count}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, f"Angle: {int(angle)} deg", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Pushup Counter (Mediapipe)", cv2.resize(frame, (960, 800)))
        if cv2.waitKey(1) & 0xFF == 27:  # ESC로 종료
            break

cap.release()
cv2.destroyAllWindows()
print(f"최종 카운트: {count}")
