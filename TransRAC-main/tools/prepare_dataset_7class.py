import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# ------------------------------------------------------
# 경로 설정
# ------------------------------------------------------
BASE = r"C:\mycla\TransRAC-main\RepCountA"
CSV_DIR = os.path.join(BASE, "annotation")
SKELETON_DIR = os.path.join(CSV_DIR, "skeleton_npz")
SAVE_BASE = os.path.join(BASE, "npz_7class")

# ------------------------------------------------------
# 클래스 alias 매핑
# ------------------------------------------------------
alias_map = {
    "squant": "squat",
    "pull_up": "pullup", "pullups": "pullup",
    "push_up": "pushup", "pushups": "pushup",
    "jump_jack": "jumpjack", "jumpjacks": "jumpjack",
    "bench_pressing": "benchpress", "benchpressing": "benchpress",
    "front_raise": "frontraise", "frontraise": "frontraise",
    "sit_ups": "situp", "situps": "situp"
}

keep_classes = ["pullup", "pushup", "jumpjack", "squat", "benchpress", "frontraise", "situp"]


# ------------------------------------------------------
# CSV 필터링
# ------------------------------------------------------
def filter_csv(csv_name):
    src = os.path.join(CSV_DIR, f"{csv_name}.csv")
    dst = os.path.join(CSV_DIR, f"{csv_name}_7class.csv")

    df = pd.read_csv(src)
    df["type"] = df["type"].astype(str).map(lambda s: alias_map.get(s.lower(), s.lower()))
    df = df[df["type"].isin(keep_classes)].reset_index(drop=True)
    df.to_csv(dst, index=False)
    print(f"✅ {csv_name}.csv → {csv_name}_7class.csv 저장 완료 ({len(df)}개 샘플)")
    return dst


# ------------------------------------------------------
# skeleton_npz 매칭 (train/valid/test 하위 폴더 탐색)
# ------------------------------------------------------
def make_split(csv_path, split_name):
    save_dir = os.path.join(SAVE_BASE, split_name)
    src_split_dir = os.path.join(SKELETON_DIR, split_name)
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"\n▶ [{split_name.upper()}] {len(df)}개 샘플 처리 중...")

    kept, skipped = 0, 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        label = str(row["type"]).lower()
        label = alias_map.get(label, label)
        video_name = str(row["name"]).replace(".mp4", "")

        # ✅ npz 경로 수정 — 클래스 하위 폴더까지 탐색
        src_npz = os.path.join(src_split_dir, label, f"{video_name}.npz")
        dst_npz = os.path.join(save_dir, f"{video_name}.npz")

        if not os.path.exists(src_npz):
            skipped += 1
            continue

        data = np.load(src_npz, allow_pickle=True)
        np.savez_compressed(dst_npz, **data)
        kept += 1

    print(f"✅ {split_name}: {kept}개 저장 완료 | 제외/누락 {skipped}개")



# ------------------------------------------------------
# 실행
# ------------------------------------------------------
if __name__ == "__main__":
    print("📦 7-Class 데이터셋 정제 시작...")

    for split in ["train", "valid", "test"]:
        csv_path = filter_csv(split)
        make_split(csv_path, split)

    print("\n🎉 모든 데이터셋 정제 완료!")
