import torch
import torch.nn as nn
import sys, os
from dataset.RepCountA_Loader import RepCountADataset
from torch.utils.data import DataLoader
import torch.optim as optim
from collections import Counter



sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=132, hidden_dim=256, num_classes=16):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_dim*2, 1)
        self.dropout = nn.Dropout(0.3)   # ✅ Dropout 추가
        self.fc = nn.Linear(hidden_dim*2, num_classes)

    def forward(self, x):  # x: [B, T, 132]
        out, _ = self.lstm(x)             
        weights = torch.softmax(self.attn(out), dim=1)  
        context = (out * weights).sum(dim=1)  
        context = self.dropout(context)   # ✅ Dropout 적용
        return self.fc(context)


# -------------------- 데이터셋 --------------------
train_dataset = RepCountADataset(
    npz_dir="./RepCountA/npz_all",
    annotation_csv="./RepCountA/annotation/train_4class.csv",
    num_frames=64
)
valid_dataset = RepCountADataset(
    npz_dir="./RepCountA/npz_all",
    annotation_csv="./RepCountA/annotation/valid_4class.csv",
    num_frames=64
)

num_classes = len(train_dataset.label_map)  # = 4
model = LSTMClassifier(input_dim=132, hidden_dim=256, num_classes=num_classes)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32)

print("Train 클래스 수:", len(train_dataset.label_map))
print("Valid 클래스 수:", len(valid_dataset.label_map))
print("클래스 매핑:", train_dataset.label_map)

# -------------------- 모델/학습 --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = len(train_dataset.label_map)   # 자동 결정
model = LSTMClassifier(input_dim=132, hidden_dim=256, num_classes=num_classes).to(device)

# Train dataset 라벨 분포로 weight 계산
labels = [y.item() for _, y in train_dataset]
counts = Counter(labels)
weights = [1.0/counts[i] for i in range(len(train_dataset.label_map))]
weights = torch.tensor(weights, dtype=torch.float).to(device)

# Weighted CrossEntropy Loss
criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(20):
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
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# -------------------- 검증 --------------------
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for X, y in valid_loader:
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

print("Validation Accuracy:", correct / total)
