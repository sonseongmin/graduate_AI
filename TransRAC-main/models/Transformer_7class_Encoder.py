import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


import torch
import torch.nn as nn
import sys, os
from torch.utils.data import DataLoader
import torch.optim as optim
from collections import Counter

# -------------------- 데이터셋 로더 임포트 --------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.RepCountA_Loader import RepCountADataset


# -------------------- 모델 정의 --------------------
class HybridLSTMTransformer(nn.Module):
    def __init__(self, input_dim=132, hidden_dim=256, num_heads=4, num_layers=2, num_classes=7, dropout=0.5):
        super().__init__()

        # ① BiLSTM: 시간 흐름(동작 순서) 학습
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # ② Transformer: 프레임 간 관계(자세 간 상호작용) 학습
        embed_dim = hidden_dim * 2  # bidirectional → output x2
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            batch_first=True,
            dropout=dropout,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ③ 분류기: 전체 시퀀스 요약
        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, x):  # x: [B, T, 132]
        lstm_out, _ = self.lstm(x)          # [B, T, hidden_dim*2]
        trans_out = self.transformer(lstm_out)  # [B, T, hidden_dim*2]
        out = trans_out.mean(dim=1)         # [B, hidden_dim*2]
        return self.fc(out)                 # [B, num_classes]


# -------------------- MAIN 실행부 --------------------
if __name__ == "__main__":
    # ✅ 7-Class 데이터셋 경로
    npz_root = r"C:\mycla\TransRAC-main\RepCountA\annotation\skeleton_npz"
    train_csv = r"C:\mycla\TransRAC-main\RepCountA\annotation\train_7class.csv"
    valid_csv = r"C:\mycla\TransRAC-main\RepCountA\annotation\valid_7class.csv"

    # ✅ 라벨맵 생성
    train_tmp = RepCountADataset(npz_dir=npz_root, annotation_csv=train_csv, num_frames=64, normalize=True)
    shared_label_map = train_tmp.label_map

    # ✅ 데이터셋 로드
    train_dataset = RepCountADataset(
        npz_dir=npz_root,
        annotation_csv=train_csv,
        num_frames=64,
        normalize=True,
        label_map=shared_label_map
    )
    valid_dataset = RepCountADataset(
        npz_dir=npz_root,
        annotation_csv=valid_csv,
        num_frames=64,
        normalize=True,
        label_map=shared_label_map
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16)

    print("클래스 매핑:", shared_label_map)

    # ✅ 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(shared_label_map)

    model = HybridLSTMTransformer(
        input_dim=132, hidden_dim=256, num_heads=4, num_layers=2,
        num_classes=num_classes
    ).to(device)

    # ✅ 클래스 불균형 가중치 계산
    labels = [y.item() for _, y in train_dataset]
    counts = Counter(labels)
    weights = [1.0 / counts[i] for i in range(num_classes)]
    weights = torch.tensor(weights, dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

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
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        # -------------------- 검증 --------------------
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in valid_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_acc = correct / total
        print(f"Validation Accuracy: {val_acc:.4f}")

        # -------------------- Early Stopping --------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wait = 0
            torch.save(model.state_dict(), r"C:\mycla\TransRAC-main\models\best_classifier_hybrid_7class.pt")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    print(f"🎯 Training complete. Best Validation Accuracy: {best_val_acc:.4f}")
