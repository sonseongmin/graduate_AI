import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from dataset.RepCountA_Loader import RepCountADataset


# ------------------------------------------------------------
# 🔥 Hybrid Decoder + Flow Encoder (pullup/squat only)
# ------------------------------------------------------------
class HybridDecoderFlow(nn.Module):
    def __init__(self, input_dim=132, hidden_dim=256, num_heads=4, num_layers=2, num_classes=7, dropout=0.4):
        super().__init__()

        # (1) 입력 임베딩
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # (2) Transformer Encoder (global feature flow)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            dropout=dropout,
            activation="gelu"
        )
        self.flow_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # (3) Transformer Decoder (기존 구조)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            dropout=dropout,
            activation="gelu"
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        # (4) learnable query token
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # (5) 두 가지 classification head
        self.fc_decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        self.fc_flow = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # pullup/squat 인덱스 (데이터셋 라벨맵 순서에 따라 변경)
        self.pullup_idx = 3
        self.squat_idx = 6

    def forward(self, x):
        # (A) 입력 임베딩
        x = self.input_proj(x)               # [B, T, hidden_dim]
        flow_mem = self.flow_encoder(x)      # global flow context [B, T, hidden_dim]

        # (B) Decoder branch
        B = x.size(0)
        query = self.query_token.expand(B, -1, -1)
        dec_out = self.decoder(query, flow_mem)  # [B, 1, hidden_dim]
        dec_feat = dec_out.squeeze(1)
        dec_logits = self.fc_decoder(dec_feat)

        # (C) Flow branch (sequence 평균 요약)
        flow_feat = flow_mem.mean(dim=1)
        flow_logits = self.fc_flow(flow_feat)

        # (D) pullup/squat 클래스에 대해서만 flow 보정
        combined_logits = dec_logits.clone()
        combined_logits[:, self.pullup_idx] = (
            0.7 * dec_logits[:, self.pullup_idx] + 0.3 * flow_logits[:, self.pullup_idx]
        )
        combined_logits[:, self.squat_idx] = (
            0.7 * dec_logits[:, self.squat_idx] + 0.3 * flow_logits[:, self.squat_idx]
        )

        return combined_logits


# ------------------------------------------------------------
# 학습 / 검증 루프
# ------------------------------------------------------------
if __name__ == "__main__":
    npz_root = r"C:\mycla\TransRAC-main\RepCountA\annotation\skeleton_npz"
    train_csv = r"C:\mycla\TransRAC-main\RepCountA\annotation\train_7class.csv"
    valid_csv = r"C:\mycla\TransRAC-main\RepCountA\annotation\valid_7class.csv"

    train_tmp = RepCountADataset(npz_dir=npz_root, annotation_csv=train_csv, num_frames=96, normalize=True)
    shared_label_map = train_tmp.label_map
    print("클래스 매핑:", shared_label_map)

    train_dataset = RepCountADataset(npz_dir=npz_root, annotation_csv=train_csv, num_frames=96, normalize=True, label_map=shared_label_map)
    valid_dataset = RepCountADataset(npz_dir=npz_root, annotation_csv=valid_csv, num_frames=96, normalize=True, label_map=shared_label_map)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(shared_label_map)

    model = HybridDecoderFlow(input_dim=132, hidden_dim=256, num_heads=4, num_layers=2,
                              num_classes=num_classes, dropout=0.4).to(device)

    # 클래스 불균형 가중치
    labels = [y.item() for _, y in train_dataset]
    counts = Counter(labels)
    total = sum(counts.values())
    weights = [total / counts[i] for i in range(num_classes)]
    weights = torch.tensor(weights, dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)

    best_val_acc = 0
    patience, wait = 5, 0

    for epoch in range(40):
        model.train()
        running_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} | Train Loss: {avg_loss:.4f}")

        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for X, y in valid_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        val_acc = correct / total
        print(f"Validation Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wait = 0
            torch.save(model.state_dict(), r"C:\mycla\TransRAC-main\models\best_hybrid_decoderflow_7class.pt")
            print(f"✅ 모델 갱신됨 (Best Acc: {best_val_acc:.4f})")
        else:
            wait += 1
            if wait >= patience:
                print("🛑 Early stopping triggered.")
                break

    print(f"\n🎯 Training complete. Best Validation Accuracy: {best_val_acc:.4f}")

    label_names = [k for k, _ in sorted(shared_label_map.items(), key=lambda x: x[1])]
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Hybrid Decoder + Flow Encoder)")
    plt.show()

    print("\n📊 Classification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=label_names))
