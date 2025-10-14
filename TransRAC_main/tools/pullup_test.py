import sys
sys.stdout.reconfigure(encoding='utf-8')


# ===============================================
# Real-Time Rep Counter (Video + Mediapipe + Transformer)
# ===============================================
import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import time

# ----------------------------
# 1. Transformer 모델 정의 (학습 때와 동일)
# ----------------------------
class STTransformer(nn.Module):
    def __init__(self, input_dim=99, embed_dim=256, num_heads=8, depth=4, max_len=2000):
        super().__init__()
        self.embed = nn.Linear(input_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        B, T, _ = x.shape
        x = self.embed(x)
        pos_embed = self.pos_embed[:, :T, :]
        x = x + pos_embed
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc(x).squeeze(1)

# ----------------------------
# 2. 모델 로드
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = STTransformer(input_dim=99, embed_dim=256, num_heads=8, depth=4, max_len=2000).to(device)
model.load_state_dict(torch.load("best_pullup_sttransformer.pt", map_location=device))
model.eval()

# ----------------------------
# 3. Mediapipe Pose 초기화
# ----------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.6, min_tracking_confidence=0.6)

# ----------------------------
# 4. 비디오 입력
# ----------------------------
video_path = r"C:\mycla\TransRAC-main\RepCountA\video\test\stu1_40.mp4"
cap = cv2.VideoCapture(video_path)

keypoints_window = []  # 최근 프레임 저장용
frame_count = 0
rep_count = 0
prev_pred = 0

# ----------------------------
# 5. 프레임 반복
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        keypoints_window.append(keypoints)
        frame_count += 1

        # 최근 50프레임만 사용
        if len(keypoints_window) > 50:
            keypoints_window.pop(0)

        # 모델 입력 형태 변환
        skel = np.stack(keypoints_window)  # (T, 33, 3)
        skel = torch.tensor(skel, dtype=torch.float32).reshape(1, skel.shape[0], -1).to(device)

        # 예측
        with torch.no_grad():
            pred = model(skel).item()

        # count 증가 감지
        if pred - prev_pred >= 0.8:  # threshold 조정 가능
            rep_count += 1
            print(f"💪 Count +1 → {rep_count}")

        prev_pred = pred

        # 시각화
        cv2.putText(frame, f"Reps: {rep_count}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

    cv2.imshow("Rep Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
pose.close()
cv2.destroyAllWindows()

print(f"🏁 최종 운동 횟수: {rep_count}")
