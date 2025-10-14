import sys
sys.stdout.reconfigure(encoding='utf-8')

import cv2
import torch
import numpy as np
import mediapipe as mp
from collections import deque
from pullup_train_final import FramewiseSTTransformer, DEVICE, MAX_LEN

# =====================================================
# MODEL LOAD
# =====================================================
MODEL_PATH = "best_pullup_framewise.pt"
model = FramewiseSTTransformer(max_len=MAX_LEN).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("✅ Model loaded successfully!")

# =====================================================
# MEDIAPIPE INIT
# =====================================================
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def normalize_skeleton(skeleton):
    mean = skeleton.mean(axis=(0, 1), keepdims=True)
    std = skeleton.std(axis=(0, 1), keepdims=True) + 1e-6
    return (skeleton - mean) / std

# =====================================================
# CAPTURE SETUP
# =====================================================
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(r"C:\mycla\TransRAC-main\RepCountA\video\test\stu1_40.mp4")

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

frames = deque(maxlen=MAX_LEN)
window = deque(maxlen=10)
rep_count = 0
curr_phase = 0
prev_phase = 0
stable_frames = 0
avg_prob = 0.0  # ← 기본값 추가 (NameError 방지)

UP_THRESHOLD = 0.6
DOWN_THRESHOLD = 0.4

print("🔥 Real-time Pull-up Counter (Press ESC to quit)")

# =====================================================
# LOOP
# =====================================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark])
        frames.append(landmarks)

        if len(frames) > 10:
            skeleton = np.array(frames)
            skeleton_norm = normalize_skeleton(skeleton)

            if len(skeleton_norm) < MAX_LEN:
                pad = np.zeros((MAX_LEN - len(skeleton_norm), 33, 3))
                skeleton_norm = np.concatenate([pad, skeleton_norm], axis=0)
            elif len(skeleton_norm) > MAX_LEN:
                skeleton_norm = skeleton_norm[-MAX_LEN:]

            x = torch.tensor(skeleton_norm, dtype=torch.float32).reshape(1, MAX_LEN, -1).to(DEVICE)

            with torch.no_grad():
                pred = model(x)
                prob = float(pred[0, -1].cpu().numpy())
                window.append(prob)

            avg_prob = np.mean(window)

            # 위상 판정
            if avg_prob > 0.7 and prev_phase == 0:
                curr_phase = 1
            elif avg_prob < 0.3 and prev_phase == 1:
                rep_count += 1
                curr_phase = 0
                print(f"✔ Rep Count +1 ({rep_count})")
            # 위상 안정도 체크
            if curr_phase == prev_phase:
                stable_frames += 1
            else:
                stable_frames = 0

            # 하강으로 10프레임 이상 안정적으로 전이되면 +1
            if prev_phase == 1 and curr_phase == 0 and stable_frames >= 3:
                rep_count += 1
                stable_frames = 0  # 중복 방지

                print(f"✔ Rep Count +1 ({rep_count})")

            prev_phase = curr_phase

            print(f"현재 확률: {avg_prob:.4f} | Phase: {curr_phase} | Count: {rep_count}")

        # 스켈레톤 시각화
        mp_drawing.draw_landmarks(
            frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        )

    # =====================================================
    # UI 표시 (항상 안전하게)
    # =====================================================
    cv2.putText(frame, f"Reps: {rep_count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, f"p={avg_prob:.2f}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Pull-up Counter", frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
