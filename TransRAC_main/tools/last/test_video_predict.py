import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import cv2
import torch
import mediapipe as mp
import numpy as np
from models.Transformer_Encoder import HybridLSTMTransformer   # ✅ 수정됨
from dataset.RepCountA_Loader import RepCountADataset

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL_PATH = r"C:\mycla\TransRAC-main\models\best_classifier_hybrid.pt"  # ✅ 오타 수정 (hybird → hybrid)
CSV_PATH = r"C:\mycla\TransRAC-main\RepCountA\annotation\valid_4class.csv"
NUM_FRAMES = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# MODEL LOAD
# --------------------------------------------------
print("모델 불러오는 중...")
model = HybridLSTMTransformer(   # ✅ 수정됨
    input_dim=132,
    hidden_dim=256,
    num_heads=4,
    num_layers=2,
    num_classes=4
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("모델 로드 완료!")

# --------------------------------------------------
# 라벨 매핑 불러오기
# --------------------------------------------------
tmp_dataset = RepCountADataset(
    npz_dir=r"C:\mycla\TransRAC-main\RepCountA\npz_all",
    annotation_csv=CSV_PATH,
    num_frames=NUM_FRAMES,
    normalize=True
)
label_map = tmp_dataset.label_map
inv_label_map = {v: k for k, v in label_map.items()}
print("클래스 매핑:", inv_label_map)

# --------------------------------------------------
# FUNCTION: extract skeleton using MediaPipe
# --------------------------------------------------
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
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            pts = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark])
            frames.append(pts)
        else:
            frames.append(np.zeros((33, 4)))  # ✅ D=4로 수정 (x,y,z,visibility)

    cap.release()
    pose.close()

    skeleton = np.array(frames, dtype=np.float32)

    # 정규화
    mean = skeleton.mean(axis=(0, 1), keepdims=True)
    std = skeleton.std(axis=(0, 1), keepdims=True) + 1e-6
    skeleton = (skeleton - mean) / std

    # pad/truncate
    if skeleton.shape[0] < num_frames:
        pad = np.zeros((num_frames - skeleton.shape[0], 33, 4))
        skeleton = np.concatenate([skeleton, pad], axis=0)
    elif skeleton.shape[0] > num_frames:
        skeleton = skeleton[:num_frames]

    x = torch.tensor(skeleton, dtype=torch.float32).reshape(1, num_frames, -1)
    return x


# --------------------------------------------------
# PREDICT FUNCTION
# --------------------------------------------------
def predict_exercise(video_path):
    print(f"\n🎥 분석 시작: {video_path}")
    x = extract_pose_from_video(video_path, num_frames=NUM_FRAMES).to(DEVICE)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        pred_label = inv_label_map[pred_class]
        confidence = probs[0, pred_class].item()

    print("\n===============================")
    print(f"예측된 운동 종류: {pred_label}")
    print(f"신뢰도: {confidence*100:.2f}%")
    print("===============================")


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    video_path = r"C:\Users\user\Desktop\project\Fitness-AI-Trainer-With-Automatic-Exercise-Recognition-and-Counting-main\Fitness-AI-Trainer-With-Automatic-Exercise-Recognition-and-Counting-main\squat_4.mp4"
    if not os.path.exists(video_path):
        print(f"파일이 존재하지 않습니다: {video_path}")
    else:
        predict_exercise(video_path)
