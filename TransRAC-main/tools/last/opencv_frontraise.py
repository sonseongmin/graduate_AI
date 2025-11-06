import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import cv2
import mediapipe as mp
import numpy as np

# --- 설정 ---
VIDEO_PATH = r"C:\mycla\TransRAC-main\RepCountA\video\train\stu4_8.mp4"  # 🎥 테스트할 영상 경로
EXERCISE = "frontraise"

# 프론트레이즈 기준 각도 임계값 (Hip–Shoulder–Elbow)
LOW_TH, HIGH_TH = 35, 80       # 팔 내렸을 때↑, 팔 들었을 때↓
DEBOUNCE_FR = 2
TARGET_RANGE = 45.0
ALPHA, MIN_VIS = 0.5, 0.60

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- 각도 계산 함수 ---
def angle3(a, b, c):
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0.0
    return np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)))

# --- 양팔 평균 어깨 각도 ---
def avg_shoulder_angle(lm):
    l = (lm[mp_pose.PoseLandmark.LEFT_HIP],
         lm[mp_pose.PoseLandmark.LEFT_SHOULDER],
         lm[mp_pose.PoseLandmark.LEFT_ELBOW])
    r = (lm[mp_pose.PoseLandmark.RIGHT_HIP],
         lm[mp_pose.PoseLandmark.RIGHT_SHOULDER],
         lm[mp_pose.PoseLandmark.RIGHT_ELBOW])
    return (angle3(*l) + angle3(*r)) / 2.0

# --- 비디오 처리 ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ 비디오를 열 수 없습니다:", VIDEO_PATH)
    exit()

count, stage = 0, None
hold_low = hold_high = 0
rep_min, rep_max = 999.0, -999.0
rep_vis_sum, rep_frames = 0.0, 0
smoothed = None

KP = [
    mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP
]

with mp_pose.Pose(model_complexity=2,
                  min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = pose.process(rgb)

        if results.pose_landmarks:
            lms = results.pose_landmarks.landmark
            raw_angle = avg_shoulder_angle(lms)
            smoothed = raw_angle if smoothed is None else (ALPHA * raw_angle + (1 - ALPHA) * smoothed)
            angle = smoothed

            vis = np.mean([lms[idx].visibility for idx in KP])

            if vis >= MIN_VIS:
                rep_min, rep_max = min(rep_min, angle), max(rep_max, angle)
                rep_vis_sum += vis
                rep_frames += 1

                hold_low = hold_low + 1 if angle < LOW_TH else 0
                hold_high = hold_high + 1 if angle > HIGH_TH else 0

                if hold_high >= DEBOUNCE_FR:
                    stage = "up"      # 팔이 들린 상태
                if stage == "up" and hold_low >= DEBOUNCE_FR:
                    stage = "down"   # 팔이 내려간 상태
                    count += 1
                    rep_min, rep_max = 999.0, -999.0
                    rep_vis_sum, rep_frames = 0.0, 0
                    hold_low = hold_high = 0

            # ✅ Mediapipe 관절 및 뼈대 표시
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # ✅ 카운트 및 각도 시각화
            cv2.putText(frame, f"{EXERCISE} Count: {count}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, f"Angle: {int(angle)} deg", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Stage: {stage if stage else '-'}", (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Front Raise Counter", cv2.resize(frame, (960, 720)))
        if cv2.waitKey(1) & 0xFF == 27:  # ESC로 종료
            break

cap.release()
cv2.destroyAllWindows()

print(f"✅ 최종 카운트: {count}")
