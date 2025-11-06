import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import cv2
import mediapipe as mp
import numpy as np

# --- 설정 ---
VIDEO_PATH = r"C:\mycla\TransRAC-main\RepCountA\video\train\stu3_56.mp4"  # 🎥 테스트 영상
EXERCISE = "situp"

ALPHA, MIN_VIS = 0.4, 0.5      # 필터링 및 최소 가시성
DEBOUNCE_FR = 4                # 상태 전환 안정화 프레임

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# --- 3D 각도 계산 ---
def angle3(a, b, c):
    a, b, c = np.array([a.x, a.y, a.z]), np.array([b.x, b.y, b.z]), np.array([c.x, c.y, c.z])
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0.0
    return np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)))


# --- 양쪽 몸통 평균 각도 ---
def avg_torso_angle(lm):
    l = (lm[mp_pose.PoseLandmark.LEFT_KNEE],
         lm[mp_pose.PoseLandmark.LEFT_HIP],
         lm[mp_pose.PoseLandmark.LEFT_SHOULDER])
    r = (lm[mp_pose.PoseLandmark.RIGHT_KNEE],
         lm[mp_pose.PoseLandmark.RIGHT_HIP],
         lm[mp_pose.PoseLandmark.RIGHT_SHOULDER])
    return (angle3(*l) + angle3(*r)) / 2.0


# --- 비디오 처리 ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ 비디오를 열 수 없습니다:", VIDEO_PATH)
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / (fps * 0.8))  # 실시간보다 살짝 느리게

count, stage = 0, "down"
prev_angle = None
trend = None   # "up" or "down"
stable_frames = 0
smoothed = None

KP = [
    mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE
]

with mp_pose.Pose(model_complexity=1,
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
            raw_angle = avg_torso_angle(lms)
            smoothed = raw_angle if smoothed is None else (ALPHA * raw_angle + (1 - ALPHA) * smoothed)
            angle = smoothed

            vis = np.mean([lms[idx].visibility for idx in KP])
            if vis < MIN_VIS:
                continue

            # --- 상대적 변화 기반 로직 ---
            if prev_angle is not None:
                diff = angle - prev_angle

                # 각도가 줄어들면 up, 늘어나면 down
                if diff < -0.5:  # 조금 줄었을 때
                    new_trend = "up"
                elif diff > 0.5:  # 조금 늘었을 때
                    new_trend = "down"
                else:
                    new_trend = trend  # 거의 변화 없으면 유지

                # 트렌드가 변할 때만 stage 업데이트
                if new_trend != trend:
                    stable_frames = 0
                    trend = new_trend
                else:
                    stable_frames += 1

                # 안정된 프레임 동안 같은 방향이면 확정
                if stable_frames >= DEBOUNCE_FR:
                    if trend == "up" and stage == "down":
                        stage = "up"
                        count += 1  # ✅ down→up 전환에서만 카운트
                    elif trend == "down" and stage == "up":
                        stage = "down"

            prev_angle = angle

            # --- 시각화 ---
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            cv2.putText(frame, f"{EXERCISE} Count: {count}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, f"Angle: {int(angle)} deg", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Stage: {stage}", (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Sit-up Counter (Auto Threshold Mode)", cv2.resize(frame, (960, 720)))
        key = cv2.waitKey(delay) & 0xFF
        if key == 27:
            break
        elif key == ord(' '):  # 스페이스바로 일시정지
            cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
print(f"✅ 최종 카운트: {count}")
