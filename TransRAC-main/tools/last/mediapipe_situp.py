import argparse, json, sys
import cv2, mediapipe as mp, numpy as np

# --------------------------------------------------
# 인자 설정
# --------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--display", action="store_true")
parser.add_argument("--out", type=str, required=False, default=None)
args = parser.parse_args()

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
EXERCISE = "situp"

# --------------------------------------------------
# 3D 각도 계산 (Knee–Hip–Shoulder)
# --------------------------------------------------
def angle3(a, b, c):
    a, b, c = np.array([a.x, a.y, a.z]), np.array([b.x, b.y, b.z]), np.array([c.x, c.y, c.z])
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0.0
    return np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)))


# --------------------------------------------------
# 양쪽 몸통의 평균 각도 계산
# --------------------------------------------------
def avg_torso_angle(lm):
    left = (lm[mp_pose.PoseLandmark.LEFT_KNEE],
            lm[mp_pose.PoseLandmark.LEFT_HIP],
            lm[mp_pose.PoseLandmark.LEFT_SHOULDER])
    right = (lm[mp_pose.PoseLandmark.RIGHT_KNEE],
             lm[mp_pose.PoseLandmark.RIGHT_HIP],
             lm[mp_pose.PoseLandmark.RIGHT_SHOULDER])
    return (angle3(*left) + angle3(*right)) / 2.0


# --------------------------------------------------
# 주요 설정값
# --------------------------------------------------
KP = [
    mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE
]
ALPHA, MIN_VIS = 0.4, 0.5
DEBOUNCE_FR = 4  # 상태 전환 안정화 프레임

# --------------------------------------------------
# 비디오 열기
# --------------------------------------------------
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    print(json.dumps({"error": f"failed to open video: {args.video}"}))
    sys.exit(2)

count, stage = 0, "down"
prev_angle = None
trend = None  # "up" or "down"
stable_frames = 0
smoothed = None
rep_qualities = []

rep_min, rep_max = 999.0, -999.0
rep_vis_sum, rep_frames = 0.0, 0

# --------------------------------------------------
# 메인 루프
# --------------------------------------------------
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

            rep_min, rep_max = min(rep_min, angle), max(rep_max, angle)
            rep_vis_sum += vis
            rep_frames += 1

            # --- 상대적 변화 기반 로직 ---
            if prev_angle is not None:
                diff = angle - prev_angle

                # 각도가 줄면 up, 늘면 down
                if diff < -0.5:
                    new_trend = "up"
                elif diff > 0.5:
                    new_trend = "down"
                else:
                    new_trend = trend  # 거의 변화 없으면 유지

                # 트렌드 변화 시 초기화
                if new_trend != trend:
                    stable_frames = 0
                    trend = new_trend
                else:
                    stable_frames += 1

                # 안정 프레임 유지 시 상태 확정
                if stable_frames >= DEBOUNCE_FR:
                    if trend == "up" and stage == "down":
                        stage = "up"
                        count += 1  # ✅ down→up 전환에서만 카운트
                    elif trend == "down" and stage == "up":
                        stage = "down"

            prev_angle = angle

        # --------------------------------------------------
        # 시각화 (옵션)
        # --------------------------------------------------
        if args.display:
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(image, f"{EXERCISE} count:{count}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(image, f"Angle: {int(angle)}°", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(image, f"Stage: {stage}", (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Sit-up Counter (Auto Threshold Mode)", cv2.resize(image, (960, 540)))
            if cv2.waitKey(1) & 0xFF == 27:
                break

cap.release()
cv2.destroyAllWindows()

# --------------------------------------------------
# 결과 계산 및 출력
# --------------------------------------------------
TARGET_RANGE = 65.0
rep_range = max(0.0, rep_max - rep_min)
coverage = min(1.0, rep_range / TARGET_RANGE)
quality = 0.5 * coverage + 0.5 * (rep_vis_sum / max(1, rep_frames))
if rep_frames > 0:
    rep_qualities.append(quality)

accuracy = 0.0 if len(rep_qualities) == 0 else float(np.mean(rep_qualities) * 100.0)
result = {"rep_count": int(count), "avg_accuracy": int(round(accuracy))}

if args.out:
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
else:
    print(json.dumps(result, ensure_ascii=False))
