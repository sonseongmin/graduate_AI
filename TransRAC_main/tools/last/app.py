import sys, os
import tempfile, subprocess, json, torch, cv2, mediapipe as mp, numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

# ✅ Python 인코딩 설정
sys.stdout.reconfigure(encoding='utf-8')#a

# ✅ 루트 경로 동적 설정 (transRAC-main 기준)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # /app/transRAC-main/tools/last
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../.."))  # /app/transRAC-main
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "TransRAC_main"))

# ✅ 내부 모듈 임포트
from models.Transformer_Encoder import HybridLSTMTransformer

# -------------------------------------------------------------------
# 기본 설정
# -------------------------------------------------------------------
app = FastAPI(title="Fitness AI Server", version="1.0.2")

SCRIPTS_DIR = os.path.join(BASE_DIR)
MODEL_PATH  = os.path.join(ROOT_DIR, "models", "best_classifier_hybrid.pt")

SCRIPTS = {
    "pushup":   os.path.join(SCRIPTS_DIR, "mediapipe_pushup.py"),
    "squat":    os.path.join(SCRIPTS_DIR, "mediapipe_squat.py"),
    "pullup":   os.path.join(SCRIPTS_DIR, "mediapipe_pullup.py"),
    "jumpjack": os.path.join(SCRIPTS_DIR, "mediapipe_jumpingjack.py"),
}

# -------------------------------------------------------------------
# ✅ CPU 전용 설정 (서버용)
# -------------------------------------------------------------------
DEVICE = "cpu"

print(f"[INFO] Using device: {DEVICE}")
print("[INFO] Loading model...")

model = HybridLSTMTransformer(
    input_dim=132, hidden_dim=256, num_heads=4, num_layers=2, num_classes=4
).to(DEVICE)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()
print("[INFO] Model loaded successfully (CPU mode).")

# ✅ 라벨맵 고정 (순서 불일치 방지)
label_map = {"pushup": 0, "squat": 1, "pullup": 2, "jumpjack": 3}
inv_label_map = {v: k for k, v in label_map.items()}

# -------------------------------------------------------------------
# ✅ 학습 시 normalize_keypoints 방식 복제
# -------------------------------------------------------------------
def normalize_keypoints(keypoints):
    """
    keypoints: [T, 33, D] (D=4 if x,y,z,visibility)
    mid-hip 중심 이동 + 어깨 거리로 스케일 정규화
    """
    # mid-hip (23,24번)
    mid_hip = (keypoints[:, 23, :3] + keypoints[:, 24, :3]) / 2
    keypoints[:, :, :3] -= mid_hip[:, None, :]

    # 어깨 거리 스케일 보정 (11,12번)
    shoulder_dist = np.linalg.norm(
        keypoints[:, 11, :3] - keypoints[:, 12, :3], axis=1
    )
    scale = shoulder_dist.mean()
    if scale > 0:
        keypoints[:, :, :3] /= scale

    return keypoints

# -------------------------------------------------------------------
# Pose 추출 함수
# -------------------------------------------------------------------
def extract_pose_from_video(video_path, num_frames=64):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_landmarks:
            pts = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark])
        else:
            pts = np.zeros((33, 4))
        frames.append(pts)

    cap.release()
    pose.close()

    skeleton = np.array(frames, dtype=np.float32)

    # ✅ 학습과 동일한 방식으로 normalize
    skeleton = normalize_keypoints(skeleton)

    # 패딩 or 자르기
    if skeleton.shape[0] < num_frames:
        pad = np.zeros((num_frames - skeleton.shape[0], 33, 4))
        skeleton = np.concatenate([skeleton, pad], axis=0)
    elif skeleton.shape[0] > num_frames:
        skeleton = skeleton[:num_frames]

    x = torch.tensor(skeleton, dtype=torch.float32).reshape(1, num_frames, -1)
    return x.to(DEVICE)

# -------------------------------------------------------------------
# 분류 함수
# -------------------------------------------------------------------
def predict_exercise(video_path):
    x = extract_pose_from_video(video_path)
    print(f"[DEBUG] Input tensor shape: {x.shape}")
    with torch.no_grad():
        outputs = model(x)
        print(f"[DEBUG] Model output shape: {outputs.shape}")
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()
    return inv_label_map[pred_class], confidence

# -------------------------------------------------------------------
# 결과 로그
# -------------------------------------------------------------------
RESULTS_DIR = os.path.join(SCRIPTS_DIR, "results")
RESULTS_LOG = os.path.join(RESULTS_DIR, "results.jsonl")

def _log_result(row: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

# -------------------------------------------------------------------
# API 엔드포인트
# -------------------------------------------------------------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        video_path = os.path.abspath(tmp.name)

    try:
        # 1️⃣ 운동 분류
        ex, conf = predict_exercise(video_path)
        print(f"[INFO] Predicted exercise: {ex} ({conf*100:.2f}%)")

        if ex not in SCRIPTS:
            raise HTTPException(status_code=400, detail=f"No script found for predicted exercise: {ex}")

        # 2️⃣ 해당 스크립트 실행
        out_fd, out_json_path = tempfile.mkstemp(prefix="mp_out_", suffix=".json")
        os.close(out_fd)

        script_path = SCRIPTS[ex]
        print(f"[DEBUG] Running script: {script_path}", flush=True)

        # ✅ 동일한 Python 환경에서 실행
        proc = subprocess.run(
            [sys.executable, script_path, "--video", video_path, "--out", out_json_path],
            capture_output=True, text=True, encoding="utf-8"
        )

        print("[DEBUG] Script stdout:", proc.stdout, flush=True)
        print("[DEBUG] Script stderr:", proc.stderr, flush=True)
        print(f"[DEBUG] Script exit code: {proc.returncode}", flush=True)

        if proc.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Script error: {proc.stderr}")

        # 3️⃣ 결과 JSON 읽기
        if not os.path.exists(out_json_path):
            raise HTTPException(status_code=500, detail="Result JSON not generated")

        with open(out_json_path, "r", encoding="utf-8") as f:
            out = json.load(f)

        rep = int(out.get("rep_count", 0))
        acc = int(out.get("avg_accuracy", 0))

        # 4️⃣ 로그 및 반환
        row = {"exercise_type": ex, "rep_count": rep, "avg_accuracy": acc}
        _log_result(row)

        return JSONResponse(content=row)

    finally:
        try:
            os.remove(video_path)
        except Exception:
            pass
