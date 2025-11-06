import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =====================================================
# CONFIG
# =====================================================
CSV_PATH = r"C:\mycla\TransRAC-main\RepCountA\annotation\train_4class.csv"
SKELETON_DIR = r"C:\mycla\TransRAC-main\RepCountA\annotation\skeleton_npz\train\pullup"
MAX_LEN = 500
BATCH_SIZE = 4
EPOCHS = 80
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================
# DATASET
# =====================================================
class PullupDataset(Dataset):
    def __init__(self, csv_path, skeleton_dir, max_len=500):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["type"] == "pullup"].reset_index(drop=True)
        self.skeleton_dir = skeleton_dir
        self.max_len = max_len

    def make_binary_label(self, total_frames, frame_points):
        """
        프레임 단위 0/1 라벨 (운동 위상)
        한 반복의 절반 전까지는 0, 이후 절반부터 다음 반복까지는 1
        """
        label = np.zeros(total_frames)
        if len(frame_points) < 2:
            return label

        for i in range(len(frame_points) - 1):
            start = int(frame_points[i])
            end = int(frame_points[i + 1])
            mid = (start + end) // 2
            label[start:mid] = 0
            label[mid:end] = 1
        return label

    def normalize(self, skeleton):
        mean = skeleton.mean(axis=(0, 1), keepdims=True)
        std = skeleton.std(axis=(0, 1), keepdims=True) + 1e-6
        return (skeleton - mean) / std

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npz_name = row["name"].replace(".mp4", ".npz")
        npz_path = os.path.join(self.skeleton_dir, npz_name)
        data = np.load(npz_path)
        key = "pose" if "pose" in data else list(data.keys())[0]
        skeleton = data[key]  # (T,33,3)
        T = skeleton.shape[0]

        skeleton = self.normalize(skeleton)
        frame_points = [float(v) for v in row[4:].values if not np.isnan(v)]
        y = self.make_binary_label(T, frame_points)
        y = torch.tensor(y, dtype=torch.float32)

        # padding / truncation
        if T < self.max_len:
            pad = np.zeros((self.max_len - T, 33, 3))
            skeleton = np.concatenate([skeleton, pad], axis=0)
            y = torch.cat([y, torch.zeros(self.max_len - T)], dim=0)
        elif T > self.max_len:
            skeleton = skeleton[:self.max_len]
            y = y[:self.max_len]

        x = torch.tensor(skeleton, dtype=torch.float32).reshape(self.max_len, -1)
        return x, y

    def __len__(self):
        return len(self.df)


# =====================================================
# MODEL (프레임별 예측용)
# =====================================================
class FramewiseSTTransformer(nn.Module):
    def __init__(self, input_dim=99, embed_dim=256, num_heads=8, depth=4, max_len=500):
        super().__init__()
        self.embed = nn.Linear(input_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        # 프레임별 출력 (T개)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()  # 0~1 구간 확률 출력
        )


    def forward(self, x):
        # x: (B, T, 99)
        B, T, _ = x.shape
        x = self.embed(x)
        x = x + self.pos_embed[:, :T, :]
        x = self.encoder(x)
        x = self.fc(x).squeeze(-1)  # (B, T)
        return x


# =====================================================
# TRAIN LOOP
# =====================================================
def train_model():
    dataset = PullupDataset(CSV_PATH, SKELETON_DIR, MAX_LEN)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = FramewiseSTTransformer(max_len=MAX_LEN).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCELoss()

    best_loss = float("inf")
    patience, wait = 10, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for x, y in tqdm(train_loader, desc=f"[Epoch {epoch}/{EPOCHS}]"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)  # (B, T)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            wait = 0
            torch.save(model.state_dict(), "best_pullup_framewise.pt")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    print(f"✅ Training finished. Best loss={best_loss:.4f}")
    return model


# =====================================================
# EVALUATE
# =====================================================
def evaluate_model(model):
    dataset = PullupDataset(CSV_PATH, SKELETON_DIR, MAX_LEN)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_mae = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            mae = torch.mean(torch.abs(pred - y)).item()
            all_mae.append(mae)

    print(f"\n📊 평균 프레임별 MAE: {np.mean(all_mae):.3f}")


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    model = train_model()
    evaluate_model(model)
