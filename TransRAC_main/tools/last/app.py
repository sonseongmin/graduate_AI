import sys, os, tempfile, subprocess, json, torch, cv2, mediapipe as mp, numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

sys.stdout.reconfigure(encoding='utf-8')

# -------------------------------------------------------------------
# FastAPI 설정
# -------------------------------------------------------------------
app = FastAPI(title="Fitness AI Server", version="0.8.0")

# -------------------------------------------------------------------
# 경로 설정 (절대 경로)
# -------------------------------------------------------------------
MODEL_PATH = "/app/TransRAC_main/models/best_classifier_hybrid.pt"
CSV_PATH   = "/app/TransRAC_main/RepCountA/annotation/valid_4class.csv"
NPZ_DIR    = "/app/TransRAC_main/RepCountA/npz_all"
SCRIPTS_DIR = "/app/TransRAC_main/tools/last"
RESULTS_DIR = "/app/TransRAC_main/tools/last/results"
RESULTS_LOG = "/app/TransRAC_main/tools/last/results/results.jsonl"

# -------------------------------------------------------------------
# 모델 및 데이터셋 로드
# -------------------------------------------------------------------
from models.Transformer_Encoder import HybridLSTMTransformer
from dataset.RepCountA_Loader import RepCountADataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 4

print("모델 불러오는 중...")
model = HybridLSTMTransformer(
    input_dim=132, hidden_dim=256, num_heads=4, num_layers=2, num_classes=NUM_CLASSES
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("모델 로드 완료!")

# 라벨 맵
tmp_dataset = RepCountADataset(
    npz_dir=NPZ_DIR,
    annotation_csv=CSV_PATH,
    num_frames=64,
    normalize=True
)
inv_label_map = {v: k for k, v in tmp_dataset.label_map.items()}

# -------------------------------------------------------------------
# Pose 추출
# -------------------------------------------------------------------
def extract_pose_from_video(video_path, num_frames=64):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
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
    mean = skeleton.mean(axis=(0, 1), keepdims=True)
    std = skeleton.std(axis=(0, 1), keepdims=True) + 1e-6
    skeleton = (skeleton - mean) / std

    if skeleton.shape[0] < num_frames:
        pad = np.zeros((num_frames - skeleton.shape[0], 33, 4))
        skeleton = np.concatenate([skeleton, pad], axis=0)
    elif skeleton.shape[0] > num_frames:
        skeleton = skeleton[:num_frames]

    x = torch.tensor(skeleton, dtype=torch.float32).reshape(1, num_frames, -1)
    return x.to(DEVICE)

# -------------------------------------------------------------------
# 예측
# -------------------------------------------------------------------
def predict_exercise(video_path):
    x = extract_pose_from_video(video_path)
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()
    return inv_label_map[pred_class], confidence

# -------------------------------------------------------------------
# 로그 기록
# -------------------------------------------------------------------
def _log_result(row: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

# -------------------------------------------------------------------
# 운동 스크립트 경로
# -------------------------------------------------------------------
SCRIPTS = {
    "pushup":   "/app/TransRAC_main/tools/last/mediapipe_pushup.py",
    "squat":    "/app/TransRAC_main/tools/last/mediapipe_squat.py",
    "pullup":   "/app/TransRAC_main/tools/last/mediapipe_pullup.py",
    "jumpjack": "/app/TransRAC_main/tools/last/mediapipe_jumpingjack.py",
}

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
        ex, conf = predict_exercise(video_path)
        print(f"예측된 운동: {ex} ({conf*100:.2f}%)")

        if ex not in SCRIPTS:
            raise HTTPException(status_code=400, detail=f"예측된 운동 스크립트 없음: {ex}")

        out_fd, out_json_path = tempfile.mkstemp(prefix="mp_out_", suffix=".json")
        os.close(out_fd)
        out_json_path = os.path.abspath(out_json_path)

        script_path = SCRIPTS[ex]
        proc = subprocess.run(
            ["python", script_path, "--video", video_path, "--out", out_json_path],
            capture_output=True, text=True, encoding="utf-8"
        )

        if proc.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Script error: {proc.stderr}")

        if not os.path.exists(out_json_path):
            raise HTTPException(status_code=500, detail="결과 JSON이 생성되지 않음")

        with open(out_json_path, "r", encoding="utf-8") as f:
            out = json.load(f)

        rep = int(out.get("rep_count", 0))
        acc = int(out.get("avg_accuracy", 0))

        row = {"exercise_type": ex, "rep_count": rep, "avg_accuracy": acc}
        _log_result(row)

        return JSONResponse(content=row)

    finally:
        try:
            os.remove(video_path)
        except Exception:
            pass
