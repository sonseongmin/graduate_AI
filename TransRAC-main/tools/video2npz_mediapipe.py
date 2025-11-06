import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

mp_pose = mp.solutions.pose

# ✅ 우리가 학습할 7개 클래스만
TARGET_CLASSES = [
    "pushup", "pullup", "squat", "jumpjack",
    "benchpress", "frontraise", "situp"
]

# ✅ CSV 내 다양한 표기를 표준화
LABEL_MAP = {
    "push_up": "pushup",
    "pushups": "pushup",
    "pull_up": "pullup",
    "pullups": "pullup",
    "squant": "squat",
    "squat": "squat",
    "jump_jack": "jumpjack",
    "jumpjacks": "jumpjack",
    "bench_pressing": "benchpress",
    "benchpressing": "benchpress",
    "front_raise": "frontraise",
    "frontraise": "frontraise",
    "situp": "situp",
}

# ✅ 무시할 클래스
IGNORE_CLASSES = ["battle_rope", "pommelhorse", "others"]


def extract_keypoints(video_path, num_frames=64):
    """Mediapipe를 사용해 비디오에서 keypoints 추출"""
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
            # ✅ visibility 포함하여 [33, 4] 구조로 저장
            pts = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in result.pose_landmarks.landmark])
        else:
            pts = np.zeros((33, 4))  # ✅ 동일한 shape 유지
        frames.append(pts)

    cap.release()
    pose.close()

    if len(frames) < num_frames:
        frames += [frames[-1]] * (num_frames - len(frames))
    return np.array(frames[:num_frames])

def process_split(csv_path, video_root, out_root):
    """train/valid/test 각각 처리"""
    df = pd.read_csv(csv_path)
    print(f"\n📂 {os.path.basename(csv_path)} 처리 중 ({len(df)}개)")

    # ✅ 라벨 정규화
    df["type"] = df["type"].str.lower().map(lambda x: LABEL_MAP.get(x, x))

    # ✅ 불필요 클래스 제거
    df = df[~df["type"].isin(IGNORE_CLASSES)]

    # ✅ 7개 클래스만 남기기
    df = df[df["type"].isin(TARGET_CLASSES)]

    print(f"→ 사용 클래스: {sorted(df['type'].unique().tolist())}")
    print(f"→ 변환할 영상 수: {len(df)}개\n")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        video_file = row["name"]
        label = row["type"]
        video_path = os.path.join(video_root, video_file)

        if not os.path.exists(video_path):
            print(f"⚠️ 영상 없음: {video_path}")
            continue

        # 저장 경로 생성
        save_dir = os.path.join(out_root, label)
        os.makedirs(save_dir, exist_ok=True)
        npz_path = os.path.join(save_dir, os.path.splitext(video_file)[0] + ".npz")

        # ✅ 이미 존재하면 건너뛰기 (기존 데이터 보호)
        if os.path.exists(npz_path):
            continue

        keypoints = extract_keypoints(video_path)
        np.savez_compressed(npz_path, keypoints=keypoints)

    print(f"✅ {os.path.basename(csv_path)} 완료!\n")


if __name__ == "__main__":
    base_dir = r"C:\mycla\TransRAC-main\RepCountA"
    csv_dir = os.path.join(base_dir, "annotation")

    for split in ["train", "valid", "test"]:
        csv_path = os.path.join(csv_dir, f"{split}.csv")
        video_root = os.path.join(base_dir, "video", split)
        out_root = os.path.join(base_dir, "annotation", "skeleton_npz", split)

        process_split(csv_path, video_root, out_root)
