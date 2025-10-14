import argparse, json, sys
import cv2, mediapipe as mp, numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--display", action="store_true")
parser.add_argument("--out", type=str, required=False, default=None)
args = parser.parse_args()

mp_pose = mp.solutions.pose
EXERCISE = "pushup"

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

def avg_elbow_angle(lm):
    l = (lm[mp_pose.PoseLandmark.LEFT_SHOULDER],
         lm[mp_pose.PoseLandmark.LEFT_ELBOW],
         lm[mp_pose.PoseLandmark.LEFT_WRIST])
    r = (lm[mp_pose.PoseLandmark.RIGHT_SHOULDER],
         lm[mp_pose.PoseLandmark.RIGHT_ELBOW],
         lm[mp_pose.PoseLandmark.RIGHT_WRIST])
    return (calculate_angle_3d(*l) + calculate_angle_3d(*r)) / 2.0

KP = [
    mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST
]

LOW_TH, HIGH_TH, DEBOUNCE_FR = 100, 130, 3
TARGET_RANGE = 80.0

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    print(json.dumps({"error": f"failed to open video: {args.video}"}))
    sys.exit(2)

count, stage = 0, None
hold_low = hold_high = 0
rep_min, rep_max = 999.0, -999.0
rep_vis_sum, rep_frames = 0.0, 0
rep_qualities = []

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
            angle = avg_elbow_angle(lms)
            vis = np.mean([lms[idx].visibility for idx in KP])

            rep_min, rep_max = min(rep_min, angle), max(rep_max, angle)
            rep_vis_sum += vis
            rep_frames += 1

            hold_low = hold_low + 1 if angle < LOW_TH else 0
            hold_high = hold_high + 1 if angle > HIGH_TH else 0

            # 상태 변화 감지
            if hold_low >= DEBOUNCE_FR:
                stage = "down"
            if stage == "down" and hold_high >= DEBOUNCE_FR:
                stage = "up"
                count += 1
                rep_range = max(0.0, rep_max - rep_min)
                coverage = min(1.0, rep_range / TARGET_RANGE)
                quality = 0.5 * coverage + 0.5 * (rep_vis_sum / max(1, rep_frames))
                rep_qualities.append(quality)
                rep_min, rep_max = 999.0, -999.0
                rep_vis_sum, rep_frames = 0.0, 0
                hold_low = hold_high = 0

        if args.display:
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(image, f"{EXERCISE} count:{count}",
                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 255), 2)
            cv2.imshow("Pushup Counter", cv2.resize(image, (960, 540)))
            if cv2.waitKey(1) & 0xFF == 27:
                break

cap.release()
cv2.destroyAllWindows()

accuracy = 0.0 if len(rep_qualities) == 0 else float(np.mean(rep_qualities) * 100.0)

# --- 결과 출력 / 저장 ---
result = {"rep_count": int(count), "avg_accuracy": int(round(accuracy))}

if args.out:
    # FastAPI에서 --out 인자가 전달되면 결과를 파일로 저장
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
else:
    # 로컬 테스트 시에는 콘솔에 출력
    print(json.dumps(result, ensure_ascii=False))
