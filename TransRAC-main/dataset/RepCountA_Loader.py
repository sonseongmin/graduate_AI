import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


# ---------------------------
# 스켈레톤 정규화 함수
# ---------------------------
def normalize_keypoints(keypoints):
    """
    keypoints: [T, 33, D] (D=4 if x,y,z,visibility)
    mid-hip 중심 이동 + 어깨 거리로 스케일 정규화
    """
    # mid-hip (23,24번)
    mid_hip = (keypoints[:, 23, :3] + keypoints[:, 24, :3]) / 2
    keypoints[:, :, :3] -= mid_hip[:, None, :]

    # 어깨 거리 스케일 보정 (11,12번)
    shoulder_dist = np.linalg.norm(
        keypoints[:, 11, :3] - keypoints[:, 12, :3], axis=1
    )
    scale = shoulder_dist.mean()
    if scale > 0:
        keypoints[:, :, :3] /= scale

    return keypoints


# ---------------------------
# Dataset 클래스
# ---------------------------
class RepCountADataset(Dataset):
    def __init__(
        self,
        npz_dir,
        annotation_csv,
        num_frames=64,
        with_visibility=True,
        label_map=None,
        alias_map=None,
        normalize=False,   # ✅ 기본값 False → LSTM/Transformer 둘 다 호환
    ):
        self.npz_dir = npz_dir
        self.num_frames = num_frames
        self.with_visibility = with_visibility
        self.normalize = normalize

        df = pd.read_csv(annotation_csv)
        df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

        # ---------------------------
        # alias 통일
        # ---------------------------
        if alias_map is None:
            alias_map = {
                "squant": "squat",
                "pull_up": "pullup",
                "pullups": "pullup",
                "push_up": "pushup",
                "pushups": "pushup",
                "jump_jack": "jumpjack",
                "jumpjacks": "jumpjack",
                "bench_pressing": "benchpressing",
                "benchpressing": "benchpress",
                "frontraise": "front_raise",
            }
        df["type"] = df["type"].astype(str).map(lambda s: alias_map.get(s, s))

        # ---------------------------
        # 필요 시 특정 클래스만 사용 (예: 4클래스)
        # ---------------------------
        keep = {"squat", "pullup", "pushup", "jumpjack"}
        df = df[df["type"].isin(keep)]

        self.rows = df[["name", "type"]].reset_index(drop=True)

        # ---------------------------
        # 라벨맵 (공용 사용 가능)
        # ---------------------------
        if label_map is None:
            classes = sorted(self.rows["type"].unique())
            self.label_map = {c: i for i, c in enumerate(classes)}
        else:
            self.label_map = label_map

        # ---------------------------
        # 샘플 리스트 구성
        # ---------------------------
        self.samples = []
        for _, r in self.rows.iterrows():
            npz_name = r["name"].replace(".mp4", ".npz")
            cls_idx = self.label_map.get(r["type"], None)
            if cls_idx is not None:
                self.samples.append((npz_name, cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        path = os.path.join(self.npz_dir, fname)
        data = np.load(path)["keypoints"]  # [T, 33, D]
        
        # ✅ normalize 옵션 적용
        if self.normalize:
            data = normalize_keypoints(data)

        # ---------------------------
        # 길이 보정 (pad / truncate)
        # ---------------------------
        T = self.num_frames
        if data.shape[0] < T:
            pad = np.zeros(
                (T - data.shape[0], data.shape[1], data.shape[2]), dtype=np.float32
            )
            data = np.concatenate([data, pad], axis=0)
        elif data.shape[0] > T:
            data = data[:T]

        # ---------------------------
        # Flatten [T, 33, D] → [T, 33*D]
        # ---------------------------
        data = data.reshape(T, -1).astype(np.float32)

        return torch.tensor(data), torch.tensor(label, dtype=torch.long)
