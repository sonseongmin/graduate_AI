import os, sys, json, argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.RepCountA_Loader import RepCountADataset


# ----------------- Transformer 모델 -----------------
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=132, num_heads=4, num_layers=2, num_classes=4):
        super().__init__()
        enc = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, batch_first=True)
        # ✅ 학습 코드랑 동일하게 "transformer" 이름 사용
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        out = self.transformer(x)      # ✅ 여기서도 transformer
        out = out.mean(dim=1)
        out = self.dropout(out)
        return self.fc(out)


# ----------------- 라벨맵 로드 -----------------
def load_or_build_label_map(npz_root, train_csv, num_frames, normalize):
    lm_path = os.path.join(os.path.dirname(train_csv), "label_map.json")
    if os.path.exists(lm_path):
        with open(lm_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        return {str(k): int(v) for k, v in label_map.items()}

    tmp = RepCountADataset(npz_root, train_csv, num_frames=num_frames, normalize=normalize)
    label_map = tmp.label_map
    with open(lm_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    return label_map


# ----------------- 평가 -----------------
def evaluate(model, loader, device, num_classes, label_map):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    acc = correct / max(1, total)
    print(f"✅ Test Accuracy: {acc:.4f}")

    try:
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
        print("\nConfusion Matrix:")
        print(cm)
        inv = {v: k for k, v in label_map.items()}
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=[inv[i] for i in range(num_classes)]))
    except Exception as e:
        print(f"[WARN] Could not print report: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="학습된 Transformer 모델 .pth 경로")
    ap.add_argument("--npz_root", default="./RepCountA/npz_all")
    ap.add_argument("--train_csv", default="./RepCountA/annotation/train_4class.csv")
    ap.add_argument("--test_csv",  default="./RepCountA/annotation/test_4class.csv")
    ap.add_argument("--num_frames", type=int, default=64)
    ap.add_argument("--normalize", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 라벨맵 맞추기
    label_map = load_or_build_label_map(args.npz_root, args.train_csv, args.num_frames, args.normalize)
    num_classes = len(label_map)

    test_ds = RepCountADataset(
        npz_dir=args.npz_root,
        annotation_csv=args.test_csv,
        num_frames=args.num_frames,
        normalize=args.normalize,
        label_map=label_map
    )
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # 모델 로드
    model = TransformerClassifier(input_dim=132, num_heads=4, num_layers=2, num_classes=num_classes)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model = model.to(device)

    # 평가
    evaluate(model, test_loader, device, num_classes, label_map)


if __name__ == "__main__":
    main()
