import sys

sys.stdout.reconfigure(encoding='utf-8')
import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp

# =======================================
# 🚀 STTransformerCount (v2 구조)
# =======================================
class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=300):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.max_len:
            position = torch.arange(0, seq_len, device=x.device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.d_model, 2, device=x.device)
                * (-np.log(10000.0) / self.d_model)
            )
            pe = torch.zeros(seq_len, self.d_model, device=x.device)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            return x + pe
        else:
            return x + self.pe[:, :seq_len]


class STTransformerCount(nn.Module):
    def __init__(self, input_dim=198, hidden_dim=128, num_heads=4, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = TemporalPositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True,
            dropout=0.1,
            dim_feedforward=2048,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc_out(x)


# =======================================
# ⚙️ Device / Model Load
# =======================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🧠 Using {device}")

model_path = r"C:\mycla\TransRAC-main\tools\st_transformer_count_v2_best.pth"
model = STTransformerCount().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"✅ Model loaded from {model_path}")

# =======================================
# 🧍 Mediapipe Pose
# =======================================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# =======================================
# 🎥 실시간 or 영상 입력
# =======================================
cap = cv2.VideoCapture(r"C:\mycla\TransRAC-main\RepCountA\video\test\stu1_30.mp4")

window_size = 30
landmark_buffer = []
pred_count = 0.0


def normalize_pose(data):
    data = np.nan_to_num(data)
    center = (data[23, :] + data[24, :]) / 2.0
    data -= center
    scale = np.linalg.norm(data[11, :] - data[12, :])
    scale = 1.0 if scale == 0 else scale
    data /= scale
    return data

# =======================================
# 🔁 실시간 처리 루프 (창 크기 + ESC 종료 + 글씨 위치 수정)
# =======================================
cv2.namedWindow("ST-Transformer Count (Realtime)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ST-Transformer Count (Realtime)", 1280, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)

    if result.pose_landmarks:
        pts = np.array(
            [[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark]
        ).astype(np.float32)
        pts = normalize_pose(pts)
        delta = np.diff(pts, axis=0, prepend=pts[:1])
        combined = np.concatenate([pts, delta], axis=-1)  # (33,6)
        landmark_buffer.append(combined.flatten())

    # 충분히 쌓이면 모델 예측
    if len(landmark_buffer) >= window_size:
        seq = np.array(landmark_buffer[-window_size:])
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(seq_tensor).item()
        pred_count = pred

    # ✅ 프레임 먼저 확대
    frame = cv2.resize(frame, (1280, 720))

    # ✅ 그 다음에 글씨를 그림 (확대된 좌표 기준)
    text = f"Predicted Count: {pred_count:.2f}"
    cv2.putText(
        frame,
        text,
        (50, 100),  # 이제 항상 화면 안쪽에 고정됨
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        3,
        cv2.LINE_AA,
    )

    cv2.imshow("ST-Transformer Count (Realtime)", frame)

    # ✅ ESC로 종료
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
pose.close()
cv2.destroyAllWindows()
