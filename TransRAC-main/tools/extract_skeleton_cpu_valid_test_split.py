import os
import sys
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob

sys.stdout.reconfigure(encoding='utf-8')

# =======================================
# ⚙️ 기본 설정
# =======================================
base_dir = r"C:\mycla\TransRAC-main\RepCountA"
video_root = os.path.join(base_dir, "video")
anno_root = os.path.join(base_dir, "annotation")

# ✅ train은 이미 완료 → valid / test만 진행
datasets = {
    "valid": os.path.join(anno_root, "valid_4class.csv"),
    "test":  os.path.join(anno_root, "test_4class.csv"),
}

target_classes = ["pushup", "pullup", "squat", "jumpjack"]
zero_threshold = 0.7  # 전체 프레임 중 70% 이상이 0이면 스킵
mp_pose = mp.solutions.pose


# =======================================
# 🧩 Skeleton 추출 함수 (CPU 전용)
# =======================================
def extract_pose_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        if w > 960:
            frame = cv2.resize(frame, (960, int(h * (960 / w))))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)

        if result.pose_landmarks:
            pts = np.array([[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark])
        else:
            pts = np.zeros((33, 3))
        frames.append(pts)

    cap.release()
    pose.close()
    return np.array(frames)


# =======================================
# 🚀 VALID / TEST 변환 루프
# =======================================
if __name__ == "__main__":
    print("\n==== Skeleton Extraction (VALID + TEST) ====\n")

    for split, csv_path in datasets.items():
        if not os.path.exists(csv_path):
            print(f"⚠️ {split} CSV 파일이 존재하지 않음 → 스킵")
            continue

        print(f"📂 {split.upper()} 데이터셋 변환 시작...")
        df = pd.read_csv(csv_path)
        df["type"] = df["type"].str.strip()
        df = df[df["type"].isin(target_classes)]

        # ✅ 저장 경로: annotation/skeleton_npz/{split}/{운동명}
        out_root = os.path.join(anno_root, "skeleton_npz", split)
        os.makedirs(out_root, exist_ok=True)

        fail_log_path = os.path.join(out_root, f"fail_list_{split}.txt")
        with open(fail_log_path, "w", encoding="utf-8") as f:
            f.write(f"==== Skeleton Generation Fail List ({split}) ====\n")

        print(f"총 {len(df)}개의 영상이 대상입니다 ({', '.join(target_classes)}).")

        for cls in target_classes:
            output_dir = os.path.join(out_root, cls)
            os.makedirs(output_dir, exist_ok=True)
            subset = df[df["type"] == cls]

            print(f"\n🎬 {split.upper()} | {cls.upper()} 변환 시작 ({len(subset)}개)...")

            for _, row in tqdm(subset.iterrows(), total=len(subset), desc=f"{split}-{cls}"):
                file = row["name"]

                # train/valid/test 폴더 내에서 탐색
                search_pattern = os.path.join(video_root, split, "**", file)
                matched_files = glob.glob(search_pattern, recursive=True)

                if not matched_files:
                    print(f"⚠️ {file} 없음 — 건너뜀")
                    with open(fail_log_path, "a", encoding="utf-8") as f:
                        f.write(f"[MISSING] {file}\n")
                    continue

                src_path = matched_files[0]
                dst_path = os.path.join(output_dir, file.replace(".mp4", ".npz"))

                if os.path.exists(dst_path):
                    continue

                try:
                    frames = extract_pose_from_video(src_path)
                    if frames.size == 0:
                        raise ValueError("빈 프레임 (frame.size == 0)")

                    zero_ratio = np.mean(frames == 0)
                    if zero_ratio > zero_threshold:
                        print(f"⚠️ {file} — {zero_ratio:.2%} zeros → 스킵")
                        with open(fail_log_path, "a", encoding="utf-8") as f:
                            f.write(f"[OCCLUDED {zero_ratio:.2%}] {src_path}\n")
                        continue

                    np.savez_compressed(dst_path, pose=frames)

                except Exception as e:
                    print(f"⚠️ {file} 처리 실패: {e}")
                    with open(fail_log_path, "a", encoding="utf-8") as f:
                        f.write(f"[ERROR] {src_path} — {e}\n")
                    continue

            print(f"✅ {split.upper()} | {cls} 완료! 저장 경로: {output_dir}")

        print(f"\n📜 실패 목록 저장됨: {fail_log_path}")

    print("\n🏁 VALID + TEST 변환 완료!")
