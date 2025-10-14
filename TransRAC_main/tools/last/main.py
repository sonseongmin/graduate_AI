import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import cv2
import torch
import numpy as np
import mediapipe as mp
from models.Transformer_Encoder import TransformerClassifier
from dataset.RepCountA_Loader import RepCountADataset

MODEL_PATH = r"C:\mycla\TransRAC-main\models\best_classifier.pt"
CSV_PATH = r"C:\mycla\TransRAC-main\RepCountA\annotation\valid_4class.csv"
NUM_FRAMES = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("모델 불러오는 중...")
model = TransformerClassifier(input_dim=132, num_heads=4, num_layers=2, num_classes=4).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("모델 로드 완료!")

tmp_dataset = RepCountADataset(npz_dir=r"C:\mycla\TransRAC-main\RepCountA\npz_all",
                               annotation_csv=CSV_PATH, num_frames=NUM_FRAMES, normalize=True)
inv_label_map = {v: k for k, v in tmp_dataset.label_map.items()}
print("클래스 매핑:", inv_label_map)


def extract_pose_from_frame(results):
    if results.pose_landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])  # ✅ visibility 제거
    else:
        return np.zeros((33, 3))


def visualize_and_predict(video_path):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    frames = []
    pred_label = "분석중..."
    confidence = 0.0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        pts = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark])
        frames.append(pts)

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 30프레임마다 한 번씩 예측 갱신
        if len(frames) >= NUM_FRAMES and frame_idx % 30 == 0:
            skeleton = np.array(frames[-NUM_FRAMES:], dtype=np.float32)
            mean = skeleton.mean(axis=(0, 1), keepdims=True)
            std = skeleton.std(axis=(0, 1), keepdims=True) + 1e-6
            skeleton = (skeleton - mean) / std
            x = torch.tensor(skeleton, dtype=torch.float32).reshape(1, NUM_FRAMES, -1).to(DEVICE)

            with torch.no_grad():
                outputs = model(x)
                probs = torch.softmax(outputs, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                pred_label = inv_label_map[pred_class]
                confidence = probs[0, pred_class].item()

        # ---------- 시각화 ----------
        cv2.putText(frame, f"Pred: {pred_label}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, f"Conf: {confidence*100:.1f}%", (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)
        cv2.imshow("실시간 운동 분류", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC 종료
            break

    cap.release()
    pose.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = r"C:\Users\user\Desktop\project\Fitness-AI-Trainer-With-Automatic-Exercise-Recognition-and-Counting-main\Fitness-AI-Trainer-With-Automatic-Exercise-Recognition-and-Counting-main\squat_1.mp4"
    visualize_and_predict(video_path)
