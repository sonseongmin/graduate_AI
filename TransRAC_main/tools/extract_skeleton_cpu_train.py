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
# ⚙️ 사용자 설정
# =======================================
video_root = r"C:\mycla\TransRAC-main\RepCountA\video"  # ✅ train/valid/test 내에 전부 있음
csv_path = r"C:\mycla\TransRAC-main\RepCountA\annotation\train_4class.csv"
out_root = r"C:\mycla\TransRAC-main\RepCountA\annotation\skeleton_npz"
fail_log_path = os.path.join(out_root, "fail_list.txt")

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

        # 너무 큰 영상은 축소 (속도 향상)
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
# 🚀 메인 실행 루프
# =======================================
if __name__ == "__main__":
    os.makedirs(out_root, exist_ok=True)
    df = pd.read_csv(csv_path)
    df["type"] = df["type"].str.strip()
    df = df[df["type"].isin(target_classes)]

    with open(fail_log_path, "w", encoding="utf-8") as f:
        f.write("==== Skeleton Generation Fail List ====\n")

    print(f"\n총 {len(df)}개의 영상이 대상입니다 ({', '.join(target_classes)}).")

    for cls in target_classes:
        output_dir = os.path.join(out_root, cls)
        os.makedirs(output_dir, exist_ok=True)
        subset = df[df["type"] == cls]

        print(f"\n🎬 {cls.upper()} 변환 시작 ({len(subset)}개)...")

        for _, row in tqdm(subset.iterrows(), total=len(subset), desc=f"{cls}"):
            file = row["name"]

            # ✅ train / valid / test 폴더 내에서 전부 탐색
            search_pattern = os.path.join(video_root, "**", file)
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

        print(f"✅ {cls} 완료! 저장 경로: {output_dir}")

    print("\n🏁 모든 변환 완료!")
    print(f"📜 실패 목록: {fail_log_path}")
