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

# -------------------- 데이터셋 로더 --------------------
from dataset.RepCountA_Loader import RepCountADataset


# -------------------- 모델 정의 --------------------
class TransformerDecoderClassifier(nn.Module):
    """
    기존 HybridLSTMTransformer를 대체하는 Decoder 기반 모델
    - 입력: skeleton sequence [B, T, 132]
    - 구조: Linear → Encoder → Decoder(Query) → FC
    """
    def __init__(self, input_dim=132, hidden_dim=256, num_heads=4, num_layers=2, num_classes=7, dropout=0.4):
        super().__init__()

        # (1) 입력 임베딩
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # (2) Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            dropout=dropout,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # (3) Transformer Decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            dropout=dropout,
            activation="gelu"
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        # (4) learnable query token (요약 벡터)
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # (5) 최종 분류기
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        # x: [B, T, 132]
        x = self.input_proj(x)                     # [B, T, hidden_dim]
        memory = self.encoder(x)                   # [B, T, hidden_dim]
        B = x.size(0)
        query = self.query_token.expand(B, -1, -1) # [B, 1, hidden_dim]
        dec_out = self.decoder(query, memory)      # [B, 1, hidden_dim]
        out = dec_out.squeeze(1)                   # [B, hidden_dim]
        return self.fc(out)                        # [B, num_classes]


# -------------------- MAIN 실행부 --------------------
if __name__ == "__main__":
    # ✅ 데이터 경로
    npz_root = r"C:\mycla\TransRAC-main\RepCountA\annotation\skeleton_npz"
    train_csv = r"C:\mycla\TransRAC-main\RepCountA\annotation\train_7class.csv"
    valid_csv = r"C:\mycla\TransRAC-main\RepCountA\annotation\valid_7class.csv"

    # ✅ 라벨맵 생성
    train_tmp = RepCountADataset(npz_dir=npz_root, annotation_csv=train_csv, num_frames=96, normalize=True)
    shared_label_map = train_tmp.label_map
    print("클래스 매핑:", shared_label_map)

    # ✅ 데이터셋 로드
    train_dataset = RepCountADataset(
        npz_dir=npz_root,
        annotation_csv=train_csv,
        num_frames=96,
        normalize=True,
        label_map=shared_label_map
    )
    valid_dataset = RepCountADataset(
        npz_dir=npz_root,
        annotation_csv=valid_csv,
        num_frames=96,
        normalize=True,
        label_map=shared_label_map
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16)

    # ✅ 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(shared_label_map)

    model = TransformerDecoderClassifier(
        input_dim=132,
        hidden_dim=256,
        num_heads=4,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.4
    ).to(device)

    # ✅ 클래스 불균형 가중치
    labels = [y.item() for _, y in train_dataset]
    counts = Counter(labels)
    total = sum(counts.values())
    weights = [total / counts[i] for i in range(num_classes)]
    weights = torch.tensor(weights, dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)

    # -------------------- 학습 루프 --------------------
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

        # -------------------- 검증 --------------------
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

        # -------------------- Early Stopping --------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wait = 0
            torch.save(model.state_dict(), r"C:\mycla\TransRAC-main\models\best_decoder_7class.pt")
            print(f"✅ 모델 갱신됨 (Best Acc: {best_val_acc:.4f})")
        else:
            wait += 1
            if wait >= patience:
                print("🛑 Early stopping triggered.")
                break

    print(f"\n🎯 Training complete. Best Validation Accuracy: {best_val_acc:.4f}")

    # -------------------- Confusion Matrix / Report --------------------
    label_names = [k for k, _ in sorted(shared_label_map.items(), key=lambda x: x[1])]
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Decoder Model)")
    plt.show()

    print("\n📊 Classification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=label_names))
