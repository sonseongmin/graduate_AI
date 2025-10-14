import argparse
import os
import sys
import numpy as np
import cv2
import torch

# 외부 의존
# pip install opencv-python mediapipe torch --extra-index-url https://download.pytorch.org/whl/cu121

# 같은 폴더에 있는 패치된 모델 로더 사용
from classify_video import model, INPUT_DIM, BATCH_FIRST  # model은 이미 weight 로드됨

# TODO: 네가 쓰는 실제 라벨 순서로 교체
CLASS_NAMES = ["pushup", "squat", "pullup", "jumpingjack"]

# ============ MediaPipe Pose ============
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
except Exception as e:
    print("[ERR] mediapipe import 실패:", e)
    print("pip install mediapipe==0.10.14 등으로 설치 필요")
    sys.exit(1)


def extract_132_from_frame(results) -> np.ndarray:
    import numpy as np
    feat = np.zeros((132,), dtype=np.float32)
    if results.pose_landmarks is None:
        return feat

    lm = results.pose_landmarks.landmark
    pts = np.array([[p.x, p.y, p.z, p.visibility] for p in lm[:33]], dtype=np.float32)  # (33,4)

    # 기준점: 좌/우 힙, 좌/우 어깨
    LHIP, RHIP, LSHO, RSHO = 23, 24, 11, 12
    hip_mid = (pts[LHIP, :3] + pts[RHIP, :3]) / 2.0
    shoulder_mid = (pts[LSHO, :3] + pts[RSHO, :3]) / 2.0
    scale = np.linalg.norm(shoulder_mid[:2] - hip_mid[:2]) + 1e-6  # 2D 스케일

    # 원점 이동 + 스케일 정규화
    xyz = pts[:, :3]
    vis = pts[:, 3:4]
    xyz = (xyz - hip_mid) / scale
    out = np.concatenate([xyz, vis], axis=1).reshape(-1)  # (132,)
    return out.astype(np.float32)


def read_video_to_features(path: str, max_len: int = 100) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"비디오 열기 실패: {path}")

    feats = []
    with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            feats.append(extract_132_from_frame(res))
    cap.release()

    if len(feats) == 0:
        raise RuntimeError("프레임을 읽지 못했음")

    X = np.stack(feats, axis=0)  # (T, 132)

    # 길이 맞추기: 초과면 균일 샘플링, 부족하면 제로 패딩
    T = len(X)
    if T > max_len:
        idx = np.linspace(0, T - 1, num=max_len).astype(int)
        X = X[idx]
    elif T < max_len:
        pad = np.zeros((max_len - T, X.shape[1]), dtype=np.float32)
        X = np.concatenate([X, pad], axis=0)

    return X  # (max_len, 132)


def run_infer(video_path: str, seq_len: int = 100, step: int = 25, conf_th: float = 0.55):
    Xfull = read_video_to_features(video_path, max_len=seq_len*5)  # 여유 있게
    votes = []
    probs_acc = []

    def infer_one(X):
        x = torch.from_numpy(X).float()
        x = x.unsqueeze(0) if BATCH_FIRST else x.unsqueeze(1)
        with torch.no_grad():
            lg = model(x)
            pr = torch.softmax(lg, dim=1).cpu().numpy()[0]
        return pr

    T = len(Xfull)
    for s in range(0, max(1, T - seq_len + 1), step):
        window = Xfull[s:s+seq_len]
        if len(window) < seq_len:
            pad = np.zeros((seq_len - len(window), Xfull.shape[1]), dtype=np.float32)
            window = np.concatenate([window, pad], axis=0)
        pr = infer_one(window)
        probs_acc.append(pr)
        votes.append(np.argmax(pr))

    probs_mean = np.mean(np.stack(probs_acc, axis=0), axis=0)
    pred_idx = int(np.argmax(probs_mean))
    pred_conf = float(probs_mean[pred_idx])
    top2 = probs_mean.argsort()[-2:][::-1]
    margin = probs_mean[top2[0]] - probs_mean[top2[1]]
    if probs_mean[top2[0]] < 0.50 or margin < 0.06 or (np.bincount(votes).max()/len(votes) < 0.60):
        pred_name = "uncertain"
    else:
        pred_name = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else str(pred_idx)

    print(f"video: {os.path.basename(video_path)}")
    print(f"exercise_type: {pred_name} (idx={pred_idx}) conf={pred_conf:.3f} votes={np.bincount(votes).tolist()}")
    for i, p in enumerate(probs_mean):
        name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i)
        print(f"  {i:02d} {name}: {p:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="", help="분류할 영상 경로")
    parser.add_argument("--seq_len", type=int, default=100)
    args = parser.parse_args()

    if args.video == "":
        # 선택 창
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            path = filedialog.askopenfilename(title="Select video", filetypes=[("Video", "*.mp4;*.avi;*.mov;*.mkv")])
            if not path:
                print("취소됨")
                sys.exit(0)
            args.video = path
        except Exception:
            print("--video 경로를 인자로 넘겨줘. 예: python run_video_classify.py --video x.mp4")
            sys.exit(1)

    run_infer(args.video, seq_len=args.seq_len)
