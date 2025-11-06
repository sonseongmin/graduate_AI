import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import numpy as np
import cv2
import mediapipe as mp
from models.Transformer_7class_Encoder import HybridLSTMTransformer  # 모델 정의 파일
from dataset.RepCountA_Loader import normalize_keypoints


# -----------------------------------------------------
# 🔧 1. 환경 설정
# -----------------------------------------------------
MODEL_PATH = r"C:\mycla\TransRAC-main\models\best_classifier_hybrid_7class.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 7개 클래스 이름 (라벨 인덱스 순서 동일)
CLASS_NAMES = ["benchpress", "frontraise", "jumpjack", "pullup", "pushup", "situp", "squat"]

mp_pose = mp.solutions.pose


# -----------------------------------------------------
# 🎬 2. Mediapipe로 비디오 → keypoints 추출
# -----------------------------------------------------
def extract_keypoints(video_path, num_frames=64):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(static_image_mode=False)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // num_frames)

    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame)
        if result.pose_landmarks:
            pts = np.array([[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark])
        else:
            pts = np.zeros((33, 3))
        frames.append(pts)

    cap.release()
    pose.close()

    # 부족한 프레임 보정
    if len(frames) < num_frames:
        frames += [frames[-1]] * (num_frames - len(frames))
    frames = np.array(frames[:num_frames])

    # 정규화 (mid-hip 중심 + 어깨 거리 스케일)
    frames = normalize_keypoints(frames)

    # Flatten [T, 33, D] → [T, 99]
    frames = frames.reshape(num_frames, -1).astype(np.float32)
    return torch.tensor(frames).unsqueeze(0)  # [1, T, 99]


# -----------------------------------------------------
# 🧠 3. 예측 함수
# -----------------------------------------------------
def predict_exercise(video_path):
    # 모델 로드
    model = HybridLSTMTransformer(
        input_dim=99, hidden_dim=256, num_heads=4, num_layers=2, num_classes=len(CLASS_NAMES)
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 비디오 → keypoints
    data = extract_keypoints(video_path).to(DEVICE)

    # 예측
    with torch.no_grad():
        outputs = model(data)
        pred_idx = outputs.argmax(dim=1).item()
        confidence = torch.softmax(outputs, dim=1)[0, pred_idx].item()

    print(f"🎯 예측 결과: {CLASS_NAMES[pred_idx]}  ({confidence*100:.2f}% 확신)")


# -----------------------------------------------------
# 🚀 4. 실행부
# -----------------------------------------------------
if __name__ == "__main__":
    test_video = r"C:\mycla\TransRAC-main\RepCountA\video\test\stu1_27.mp4"  # 🔸 테스트할 영상 경로 수정
    predict_exercise(test_video)
