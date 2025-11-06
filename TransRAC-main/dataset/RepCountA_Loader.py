import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


# ---------------------------
# 스켈레톤 정규화 함수
# ---------------------------
def normalize_keypoints(keypoints):
    """
    keypoints: [T, 33, D]
    mid-hip 중심 정렬 + 어깨 거리로 스케일 정규화
    """
    mid_hip = (keypoints[:, 23, :3] + keypoints[:, 24, :3]) / 2
    keypoints[:, :, :3] -= mid_hip[:, None, :]

    shoulder_dist = np.linalg.norm(
        keypoints[:, 11, :3] - keypoints[:, 12, :3], axis=1
    )
    scale = shoulder_dist.mean()
    if scale > 0:
        keypoints[:, :, :3] /= scale
    return keypoints


# ---------------------------
# 7-Class Dataset 정의
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
        normalize=False,
    ):
        self.npz_dir = npz_dir
        self.num_frames = num_frames
        self.with_visibility = with_visibility
        self.normalize = normalize

        # ---------------------------
        # CSV 로드
        # ---------------------------
        df = pd.read_csv(annotation_csv)
        df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

        # ---------------------------
        # 클래스 이름 통일 (alias 처리)
        # ---------------------------
        if alias_map is None:
            alias_map = {
                "squant": "squat",
                "pull_up": "pullup", "pullups": "pullup",
                "push_up": "pushup", "pushups": "pushup",
                "jump_jack": "jumpjack", "jumpjacks": "jumpjack",
                "bench_pressing": "benchpress", "benchpressing": "benchpress",
                "front_raise": "frontraise", "frontraise": "frontraise",
                "situps": "situp", "sit_ups": "situp"
            }

        df["type"] = df["type"].astype(str).map(lambda s: alias_map.get(s.lower(), s.lower()))

        # ---------------------------
        # 7개 클래스만 유지
        # ---------------------------
        keep_classes = {"squat", "pullup", "pushup", "jumpjack", "benchpress", "frontraise", "situp"}
        df = df[df["type"].isin(keep_classes)].reset_index(drop=True)

        self.rows = df[["name", "type"]].reset_index(drop=True)

        # ---------------------------
        # 라벨 매핑
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

        # train / valid / test 및 클래스 하위 폴더 자동 탐색
        possible_dirs = [
            self.npz_dir,
            os.path.join(self.npz_dir, "train"),
            os.path.join(self.npz_dir, "valid"),
            os.path.join(self.npz_dir, "test"),
        ]

        path = None
        for base_dir in possible_dirs:
            # 클래스별 하위폴더도 함께 탐색
            if not os.path.exists(base_dir):
                continue
            for cls_name in os.listdir(base_dir):
                cls_dir = os.path.join(base_dir, cls_name)
                if not os.path.isdir(cls_dir):
                    continue
                candidate = os.path.join(cls_dir, fname)
                if os.path.exists(candidate):
                    path = candidate
                    break
            if path:
                break

        if path is None:
            raise FileNotFoundError(f"{fname} not found in any class folder under {self.npz_dir}")

        # ✅ npz 내부 키 자동 탐색
        npz_data = np.load(path)
        if "keypoints" in npz_data:
            data = npz_data["keypoints"]
        elif "arr_0" in npz_data:
            data = npz_data["arr_0"]
        elif "pose" in npz_data:
            data = npz_data["pose"]
        else:
            raise KeyError(f"{path} 내부에 keypoints / arr_0 / pose 키가 없습니다: {npz_data.files}")

        if self.normalize:
            data = normalize_keypoints(data)

        T = self.num_frames
        if data.shape[0] < T:
            pad = np.zeros((T - data.shape[0], data.shape[1], data.shape[2]), dtype=np.float32)
            data = np.concatenate([data, pad], axis=0)
        elif data.shape[0] > T:
            data = data[:T]

        data = data.reshape(T, -1).astype(np.float32)
        return torch.tensor(data), torch.tensor(label, dtype=torch.long)
