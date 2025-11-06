import argparse, json, sys
import cv2, mediapipe as mp
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--display", action="store_true")
parser.add_argument("--out", type=str, required=False, default=None)
args = parser.parse_args()

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
EXERCISE = "jumpingjack"

def euclid(a, b):
    return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)

KP = [
    mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE
]

OPEN_TH, CLOSE_TH, DEBOUNCE_FR = 0.70, 0.60, 1
TARGET_RANGE = 1.0

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    print(json.dumps({"error": f"failed to open video: {args.video}"}))
    sys.exit(2)

count, stage = 0, None
hold_open = hold_close = 0
rep_min, rep_max = 1e9, -1e9
rep_vis_sum, rep_frames = 0.0, 0
rep_qualities = []

with mp_pose.Pose(model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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

            openness = 0.5 * hand_height_norm + 0.5 * feet_spread_norm
            vis = np.mean([lms[idx].visibility for idx in KP])

            rep_min, rep_max = min(rep_min, openness), max(rep_max, openness)
            rep_vis_sum += vis
            rep_frames += 1

            hold_open = hold_open + 1 if openness > OPEN_TH else 0
            hold_close = hold_close + 1 if openness < CLOSE_TH else 0

            # --- 수정된 로직: open → closed 시점에 카운트 ---
            if hold_open >= DEBOUNCE_FR:
                stage = "open"
            if stage == "open" and hold_close >= DEBOUNCE_FR:
                stage = "closed"
                count += 1  # ✅ 착지 순간 카운트
                rep_range = max(0.0, rep_max - rep_min)
                coverage = min(1.0, rep_range / TARGET_RANGE)
                quality = 0.5 * coverage + 0.5 * (rep_vis_sum / max(1, rep_frames))
                rep_qualities.append(quality)
                rep_min, rep_max = 1e9, -1e9
                rep_vis_sum, rep_frames = 0.0, 0
                hold_open = hold_close = 0

            print(f"open={openness:.2f} vis={vis:.2f} stage={stage} count={count}")

        if args.display:
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(image, f"{EXERCISE} count:{count}", (30,60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            cv2.imshow("Jumping Jack Counter (Landing Count)", cv2.resize(image, (960,540)))
            if cv2.waitKey(1) & 0xFF == 27:
                break

cap.release()
cv2.destroyAllWindows()
accuracy = 0.0 if len(rep_qualities) == 0 else float(np.mean(rep_qualities) * 100.0)

# --- 결과 출력 / 저장 ---
result = {"rep_count": int(count), "avg_accuracy": int(round(accuracy))}

if args.out:
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
else:
    print(json.dumps(result, ensure_ascii=False))
