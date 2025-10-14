import os, sys, json, time, argparse, collections
import numpy as np
import cv2
import torch
import torch.nn as nn

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ----------------- 모델 (학습과 동일한 이름/구조) -----------------
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=132, num_heads=4, num_layers=2, num_classes=4, dropout=0.3):
        super().__init__()
        enc = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):  # x: [B, T, 132]
        out = self.transformer(x)      # [B, T, 132]
        out = out.mean(dim=1)          # [B, 132]
        out = self.dropout(out)
        return self.fc(out)            # [B, C]

# ----------------- 스켈레톤 정규화 -----------------
def normalize_keypoints(keypoints):  # keypoints: [T,33,4]
    mid_hip = (keypoints[:, 23, :3] + keypoints[:, 24, :3]) / 2
    keypoints[:, :, :3] -= mid_hip[:, None, :]
    shoulder_dist = np.linalg.norm(keypoints[:, 11, :3] - keypoints[:, 12, :3], axis=1)
    scale = shoulder_dist.mean()
    if scale > 0:
        keypoints[:, :, :3] /= scale
    return keypoints

# ----------------- MediaPipe Pose -----------------
def build_pose():
    import mediapipe as mp
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return mp, pose

# ----------------- 라벨맵 로드 -----------------
def load_label_map(train_csv_dir="./RepCountA/annotation"):
    lm_path = os.path.join(train_csv_dir, "label_map.json")
    if not os.path.exists(lm_path):
        raise FileNotFoundError(f"label_map.json이 없습니다: {lm_path}")
    with open(lm_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    inv_map = {int(v): k for k, v in label_map.items()}
    return label_map, inv_map

# ----------------- 메인 루프 -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="./models/best_transformer.pth", help="학습된 Transformer 가중치 경로")
    ap.add_argument("--camera", type=int, default=0, help="웹캠 인덱스 (기본 0)")
    ap.add_argument("--num_frames", type=int, default=64, help="시퀀스 길이")
    ap.add_argument("--normalize", action="store_true", help="normalize 적용 여부")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--label_dir", default="./RepCountA/annotation", help="label_map.json 위치")
    ap.add_argument("--threshold", type=float, default=0.50, help="표시 임계 확신도")
    ap.add_argument("--avg_window", type=int, default=8, help="최근 N프레임 softmax 평균으로 스무딩")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 라벨맵
    label_map, inv_label_map = load_label_map(args.label_dir)
    num_classes = len(label_map)

    # 모델 로드
    model = TransformerClassifier(
        input_dim=132, num_heads=args.heads, num_layers=args.layers,
        num_classes=num_classes, dropout=0.3
    )
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device).eval()
    print(f"✅ 모델 로드 완료: {args.ckpt}, 클래스 수={num_classes}")

    # MediaPipe Pose
    mp, pose = build_pose()
    print("✅ MediaPipe Pose 초기화 완료")

    # 비디오 캡처 (DirectShow 강제)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"❌ 웹캠 열기 실패 (camera index={args.camera})")
        return
    else:
        print(f"✅ 웹캠 연결 성공 (camera index={args.camera})")

    T = args.num_frames
    seq_buf = collections.deque(maxlen=T)
    prob_buf = collections.deque(maxlen=args.avg_window)

    fps_t0 = time.time()
    fps_cnt = 0
    fps_disp = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ 프레임 읽기 실패, 루프 종료")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                kp = np.array([[p.x, p.y, p.z, p.visibility] for p in lm], dtype=np.float32)
                if kp.shape[0] != 33:
                    kp_pad = np.zeros((33, 4), dtype=np.float32)
                    n = min(kp.shape[0], 33)
                    kp_pad[:n] = kp[:n]
                    kp = kp_pad
            else:
                kp = np.zeros((33, 4), dtype=np.float32)

            seq_buf.append(kp)

            if len(seq_buf) == T:
                seq = np.stack(seq_buf, axis=0)
                if args.normalize:
                    seq = normalize_keypoints(seq.copy())
                x = seq.reshape(T, -1)[None, ...]
                x = torch.tensor(x, dtype=torch.float32, device=device)

                with torch.no_grad():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                prob_buf.append(probs)

                probs_avg = np.mean(np.stack(prob_buf, axis=0), axis=0)
                pred_idx = int(np.argmax(probs_avg))
                pred_name = inv_label_map.get(pred_idx, str(pred_idx))
                pred_conf = float(probs_avg[pred_idx])

                h, w = frame.shape[:2]
                cv2.rectangle(frame, (10, 10), (10 + int(w*0.4), 60), (0, 0, 0), -1)
                text = f"{pred_name}  {pred_conf*100:.1f}%"
                color = (0, 255, 0) if pred_conf >= args.threshold else (0, 200, 255)
                cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            fps_cnt += 1
            if fps_cnt >= 10:
                now = time.time()
                fps_disp = fps_cnt / (now - fps_t0 + 1e-6)
                fps_t0 = now
                fps_cnt = 0
            cv2.putText(frame, f"FPS: {fps_disp:.1f}", (10, frame.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow("Exercise Classification (Transformer + MediaPipe)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                print("👋 ESC/Q 눌러서 종료")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
