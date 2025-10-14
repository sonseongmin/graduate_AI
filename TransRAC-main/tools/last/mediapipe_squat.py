import argparse, json, sys, os
import cv2, mediapipe as mp, numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--display", action="store_true")
args = parser.parse_args()
OUT_PATH = os.path.abspath(args.out)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump({"rep_count": 0, "avg_accuracy": 0}, f, ensure_ascii=False)

mp_pose = mp.solutions.pose
EXERCISE = "squat"

def angle3(a,b,c):
    a = np.array([a.x,a.y,a.z]); b = np.array([b.x,b.y,b.z]); c = np.array([c.x,c.y,c.z])
    ba, bc = a-b, c-b; denom = np.linalg.norm(ba)*np.linalg.norm(bc)
    if denom == 0: return 0.0
    return np.degrees(np.arccos(np.clip(np.dot(ba,bc)/denom, -1.0, 1.0)))

def knee_angle(lm):
    l = (lm[mp_pose.PoseLandmark.LEFT_HIP], lm[mp_pose.PoseLandmark.LEFT_KNEE], lm[mp_pose.PoseLandmark.LEFT_ANKLE])
    r = (lm[mp_pose.PoseLandmark.RIGHT_HIP], lm[mp_pose.PoseLandmark.RIGHT_KNEE], lm[mp_pose.PoseLandmark.RIGHT_ANKLE])
    return (angle3(*l) + angle3(*r)) / 2.0

KP = [
    mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE
]

LOW_TH, HIGH_TH, DEBOUNCE_FR = 90, 120, 3
TARGET_RANGE = 80.0

# 시작 시 최소 JSON 기록해 두기
_init_payload = {"rep_count": 0, "avg_accuracy": 0}
try:
    with open(OUT_PATH, "w", encoding="utf-8") as _f:
        json.dump(_init_payload, _f, ensure_ascii=False)
except Exception:
    pass

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    try:
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump({"rep_count": 0, "avg_accuracy": 0}, f, ensure_ascii=False)
    except Exception:
        pass

count, stage = 0, None
hold_low = hold_high = 0
rep_min, rep_max = 999.0, -999.0
rep_vis_sum, rep_frames = 0.0, 0
rep_qualities = []

try:
    with mp_pose.Pose(model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ok, frame = cap.read()
            if not ok: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)

            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark
                angle = knee_angle(lms)
                vis = np.mean([lms[idx].visibility for idx in KP])

                rep_min, rep_max = min(rep_min, angle), max(rep_max, angle)
                rep_vis_sum += vis; rep_frames += 1

                hold_low = hold_low + 1 if angle < LOW_TH else 0
                hold_high = hold_high + 1 if angle > HIGH_TH else 0
                if hold_low >= DEBOUNCE_FR: stage = "down"
                if stage == "down" and hold_high >= DEBOUNCE_FR:
                    stage = "up"; count += 1
                    rep_range = max(0.0, rep_max - rep_min)
                    coverage = min(1.0, rep_range / TARGET_RANGE)
                    quality = 0.5 * coverage + 0.5 * (rep_vis_sum / max(1, rep_frames))
                    rep_qualities.append(quality)
                    rep_min, rep_max = 999.0, -999.0
                    rep_vis_sum, rep_frames = 0.0, 0
                    hold_low = hold_high = 0

            if args.display:
                image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.putText(image, f"{EXERCISE} count:{count}", (30,60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                cv2.imshow("Squat Counter", cv2.resize(image, (960,540)))
                if cv2.waitKey(1) & 0xFF == 27: break
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()

    # rep_count와 avg_accuracy 계산
    accuracy = 0.0 if len(rep_qualities) == 0 else float(np.mean(rep_qualities) * 100.0)
    result = {
        "exercise_type": EXERCISE,
        "rep_count": int(count),
        "avg_accuracy": int(round(accuracy))
    }

    # 디버그 출력
    print("=== DEBUG: FINAL RESULT ===")
    print(result)
    print("OUT_PATH:", OUT_PATH)
    print("===========================")

    # JSON 파일에 최종 덮어쓰기
    import tempfile, shutil
    _dir = os.path.dirname(OUT_PATH) or "."
    _fd, _tmp = tempfile.mkstemp(prefix="mp_out_", suffix=".json", dir=_dir)
    os.close(_fd)
    try:
        with open(_tmp, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        shutil.move(_tmp, OUT_PATH)
    except Exception as e:
        print("JSON write error:", e)
        try:
            with open(OUT_PATH, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception:
            pass