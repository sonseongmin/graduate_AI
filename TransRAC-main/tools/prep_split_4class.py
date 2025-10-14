import pandas as pd
import os

# ---------------------------
# 공통 alias 매핑
# ---------------------------
alias_map = {
    "squant": "squat",
    "pull_up": "pullup", "pullups": "pullup",
    "push_up": "pushup", "pushups": "pushup",
    "jump_jack": "jumpjack", "jumpjacks": "jumpjack",
    "bench_pressing": "benchpressing", "benchpressing": "benchpress",
    "frontraise": "front_raise",
}

# ---------------------------
# 4개 클래스만 유지
# ---------------------------
keep = {"squat", "pullup", "pushup", "jumpjack"}

# ---------------------------
# 함수 정의
# ---------------------------
def process_csv(csv_path, save_path):
    df = pd.read_csv(csv_path)
    # alias 통일
    df["type"] = df["type"].astype(str).map(lambda s: alias_map.get(s, s))
    # 4개 클래스만 유지
    df = df[df["type"].isin(keep)].reset_index(drop=True)
    df.to_csv(save_path, index=False)
    print(f"✅ {csv_path} → {save_path} 저장 완료 ({len(df)} samples)")


if __name__ == "__main__":
    base = "./RepCountA/annotation"

    process_csv(os.path.join(base, "train.csv"), os.path.join(base, "train_4class.csv"))
    process_csv(os.path.join(base, "valid.csv"), os.path.join(base, "valid_4class.csv"))
    process_csv(os.path.join(base, "test.csv"),  os.path.join(base, "test_4class.csv"))
