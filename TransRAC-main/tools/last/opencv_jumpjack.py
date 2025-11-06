import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import cv2
import mediapipe as mp
import numpy as np

# --- 설정 ---
VIDEO_PATH = r"C:\Users\user\Downloads\back_jumpjack.mp4"  # 🎥 영상 경로
EXERCISE = "jumpingjack"

# 임계값 (동작 인식 기준)
OPEN_TH, CLOSE_TH = 0.70, 0.60
DEBOUNCE_FR = 1
TARGET_RANGE = 1.0
ALPHA, MIN_VIS = 0.5, 0.60

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- 유틸 ---
def euclid(a, b):
    return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)

KP = [
    mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE
]

# --- 비디오 처리 ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ 비디오를 열 수 없습니다:", VIDEO_PATH)
    exit()

count, stage = 0, None
hold_open = hold_close = 0
rep_min, rep_max = 1e9, -1e9
rep_vis_sum, rep_frames = 0.0, 0
smoothed = None

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
            l_sh, r_sh = lms[mp_pose.PoseLandmark.LEFT_SHOULDER], lms[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            l_hip, r_hip = lms[mp_pose.PoseLandmark.LEFT_HIP], lms[mp_pose.PoseLandmark.RIGHT_HIP]
            l_wri, r_wri = lms[mp_pose.PoseLandmark.LEFT_WRIST], lms[mp_pose.PoseLandmark.RIGHT_WRIST]
            l_ank, r_ank = lms[mp_pose.PoseLandmark.LEFT_ANKLE], lms[mp_pose.PoseLandmark.RIGHT_ANKLE]

            hip_width = euclid(l_hip, r_hip) + 1e-6
            ankle_dist = euclid(l_ank, r_ank)

            shoulder_y = 0.5 * (l_sh.y + r_sh.y)
            hip_y = 0.5 * (l_hip.y + r_hip.y)
            wrist_y = 0.5 * (l_wri.y + r_wri.y)
            denom = max(1e-6, hip_y - shoulder_y)
            hand_height_norm = np.clip((shoulder_y - wrist_y) / denom, 0.0, 1.0)

            feet_spread_norm = np.clip((ankle_dist / hip_width), 0.0, 1.5)
            feet_spread_norm = np.clip(feet_spread_norm / 1.0, 0.0, 1.0)

            openness_raw = 0.5 * hand_height_norm + 0.5 * feet_spread_norm
            smoothed = openness_raw if smoothed is None else (ALPHA * openness_raw + (1 - ALPHA) * smoothed)
            openness = smoothed

            vis = np.mean([lms[idx].visibility for idx in KP])

            if vis >= MIN_VIS:
                rep_min, rep_max = min(rep_min, openness), max(rep_max, openness)
                rep_vis_sum += vis
                rep_frames += 1

                # 열림/닫힘 상태 체크
                hold_open = hold_open + 1 if openness > OPEN_TH else 0
                hold_close = hold_close + 1 if openness < CLOSE_TH else 0

                # --- 로직 변경: open → closed 시 카운트 ---
                if hold_open >= DEBOUNCE_FR:
                    stage = "open"
                if stage == "open" and hold_close >= DEBOUNCE_FR:
                    stage = "closed"
                    count += 1  # ✅ 착지 순간 카운트 발생
                    rep_min, rep_max = 1e9, -1e9
                    rep_vis_sum, rep_frames = 0.0, 0
                    hold_open = hold_close = 0

            # ✅ 관절 및 뼈대 시각화
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # ✅ 텍스트 시각화 (크게 표시)
            cv2.putText(frame, f"Count: {count}", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4)
            cv2.putText(frame, f"Openness: {openness:.2f}", (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
            cv2.putText(frame, f"Stage: {stage if stage else '-'}", (30, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)

        cv2.imshow("Jumping Jack Counter (Landing Count)", cv2.resize(frame, (960, 720)))
        if cv2.waitKey(1) & 0xFF == 27:  # ESC로 종료
            break

cap.release()
cv2.destroyAllWindows()
print(f"✅ 최종 카운트: {count}")
