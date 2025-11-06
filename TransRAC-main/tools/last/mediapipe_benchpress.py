import argparse, json, sys
import cv2, mediapipe as mp, numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--display", action="store_true")
parser.add_argument("--out", type=str, required=False, default=None)
args = parser.parse_args()

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
EXERCISE = "benchpress"

# --------------------------------------------------
# 팔꿈치 각도 계산 함수
# --------------------------------------------------
def angle3(a, b, c):
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0.0
    return np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)))


# --------------------------------------------------
# 양팔 평균 팔꿈치 각도 계산
# --------------------------------------------------
def avg_elbow_angle(lm):
    left = (lm[mp_pose.PoseLandmark.LEFT_SHOULDER],
            lm[mp_pose.PoseLandmark.LEFT_ELBOW],
            lm[mp_pose.PoseLandmark.LEFT_WRIST])
    right = (lm[mp_pose.PoseLandmark.RIGHT_SHOULDER],
             lm[mp_pose.PoseLandmark.RIGHT_ELBOW],
             lm[mp_pose.PoseLandmark.RIGHT_WRIST])
    return (angle3(*left) + angle3(*right)) / 2.0


# --------------------------------------------------
# 주요 설정값
# --------------------------------------------------
KP = [
    mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST
]

# 벤치프레스 기준: 팔이 구부러졌을 때 ↓, 펴졌을 때 ↑
LOW_TH, HIGH_TH, DEBOUNCE_FR = 100, 115, 2
TARGET_RANGE = 80.0
ALPHA, MIN_VIS = 0.5, 0.6


# --------------------------------------------------
# 비디오 열기
# --------------------------------------------------
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    print(json.dumps({"error": f"failed to open video: {args.video}"}))
    sys.exit(2)

count, stage = 0, None
hold_low = hold_high = 0
rep_min, rep_max = 999.0, -999.0
rep_vis_sum, rep_frames = 0.0, 0
rep_qualities = []
smoothed = None


# --------------------------------------------------
# 메인 루프
# --------------------------------------------------
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
            raw_angle = avg_elbow_angle(lms)
            smoothed = raw_angle if smoothed is None else (ALPHA * raw_angle + (1 - ALPHA) * smoothed)
            angle = smoothed

            vis = np.mean([lms[idx].visibility for idx in KP])
            if vis >= MIN_VIS:
                rep_min, rep_max = min(rep_min, angle), max(rep_max, angle)
                rep_vis_sum += vis
                rep_frames += 1

                hold_low = hold_low + 1 if angle < LOW_TH else 0
                hold_high = hold_high + 1 if angle > HIGH_TH else 0

                if hold_low >= DEBOUNCE_FR:
                    stage = "down"  # 바벨이 내려간 상태
                if stage == "down" and hold_high >= DEBOUNCE_FR:
                    stage = "up"    # 바벨을 밀어올림
                    count += 1

                    rep_range = max(0.0, rep_max - rep_min)
                    coverage = min(1.0, rep_range / TARGET_RANGE)
                    quality = 0.5 * coverage + 0.5 * (rep_vis_sum / max(1, rep_frames))
                    rep_qualities.append(quality)

                    rep_min, rep_max = 999.0, -999.0
                    rep_vis_sum, rep_frames = 0.0, 0
                    hold_low = hold_high = 0

        # --------------------------------------------------
        # 시각화
        # --------------------------------------------------
        if args.display:
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(image, f"{EXERCISE} count:{count}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(image, f"Angle: {int(angle)}°", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(image, f"Stage: {stage if stage else '-'}", (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Bench Press Counter", cv2.resize(image, (960, 540)))
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

cap.release()
cv2.destroyAllWindows()

# --------------------------------------------------
# 결과 계산 및 출력
# --------------------------------------------------
accuracy = 0.0 if len(rep_qualities) == 0 else float(np.mean(rep_qualities) * 100.0)
result = {"rep_count": int(count), "avg_accuracy": int(round(accuracy))}

if args.out:
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
else:
    print(json.dumps(result, ensure_ascii=False))
