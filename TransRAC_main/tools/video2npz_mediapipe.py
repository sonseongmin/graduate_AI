import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

mp_pose = mp.solutions.pose

def extract_keypoints_from_video(video_path, num_frames=64, include_visibility=True):
    """
    비디오에서 Mediapipe pose keypoints (x, y, z, [visibility]) 추출
    :param video_path: mp4 파일 경로
    :param num_frames: 샘플링할 프레임 수 (모든 영상을 동일 길이로 맞춤)
    :param include_visibility: True면 visibility까지 포함 ([x,y,z,v]) / False면 [x,y,z]만
    :return: numpy array [T, K, D] (T=num_frames, K=관절 개수=33, D=3 또는 4)
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 샘플링할 프레임 인덱스 고르기
    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    keypoints_seq = []

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if i not in frame_indices:
            continue

        # BGR → RGB 변환
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            if include_visibility:
                # (x,y,z,visibility)
                keypoints = np.array(
                    [(lm.x, lm.y, lm.z, lm.visibility) for lm in results.pose_landmarks.landmark],
                    dtype=np.float32
                )
            else:
                # (x,y,z)
                keypoints = np.array(
                    [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark],
                    dtype=np.float32
                )
        else:
            dim = 4 if include_visibility else 3
            keypoints = np.zeros((33, dim), dtype=np.float32)

        keypoints_seq.append(keypoints)

    cap.release()
    pose.close()

    keypoints_seq = np.array(keypoints_seq)  # [T, 33, D]

    # 부족하면 패딩
    if keypoints_seq.shape[0] < num_frames:
        pad_len = num_frames - keypoints_seq.shape[0]
        dim = 4 if include_visibility else 3
        pad = np.zeros((pad_len, 33, dim), dtype=np.float32)
        keypoints_seq = np.concatenate([keypoints_seq, pad], axis=0)

    return keypoints_seq


def process_dataset(input_dir, output_dir, num_frames=64, include_visibility=True):
    """
    폴더 내 모든 mp4 영상을 Mediapipe keypoints npz로 변환
    :param input_dir: mp4 영상들이 들어있는 폴더
    :param output_dir: npz 파일 저장 폴더
    :param num_frames: 샘플링할 프레임 수
    :param include_visibility: True면 (x,y,z,v), False면 (x,y,z)
    """
    os.makedirs(output_dir, exist_ok=True)
    video_files = [f for f in os.listdir(input_dir) if f.endswith(".mp4")]

    for vf in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(input_dir, vf)
        keypoints_seq = extract_keypoints_from_video(video_path, num_frames, include_visibility)

        # npz로 저장
        save_path = os.path.join(output_dir, vf.replace(".mp4", ".npz"))
        np.savez_compressed(save_path, keypoints=keypoints_seq)


if __name__ == "__main__":
    # 경로를 RepCountA 기준으로 수정
    input_dir = "./RepCountA/video/train"
    output_dir = "./RepCountA/npz/train"
    process_dataset(input_dir, output_dir, num_frames=64, include_visibility=True)

    input_dir = "./RepCountA/video/valid"
    output_dir = "./RepCountA/npz/valid"
    process_dataset(input_dir, output_dir, num_frames=64, include_visibility=True)

    input_dir = "./RepCountA/video/test"
    output_dir = "./RepCountA/npz/test"
    process_dataset(input_dir, output_dir, num_frames=64, include_visibility=True)
