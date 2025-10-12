# processor.py
import cv2
import numpy as np
import mediapipe as mp
import time

mp_pose = mp.solutions.pose

# ======================
# 1) Pose 모델 전역 1회 로딩
# ======================
_pose_model = None

def analyze_video_init():
    """서버 시작 시 딱 한 번 Pose 모델을 초기화"""
    global _pose_model
    if _pose_model is None:
        _pose_model = mp_pose.Pose(static_image_mode=False, model_complexity=1)
    return _pose_model


# ======================
# 2) 각종 보조 함수들
# ======================
def _angle_3d(a, b, c):
    ba = a - b
    bc = c - b
    if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
        return None
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))

def _get(landmarks, idx):
    lm = landmarks[idx]
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

def _pushup_angle(landmarks):
    l_sh, l_el, l_wr = _get(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER), _get(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW), _get(landmarks, mp_pose.PoseLandmark.LEFT_WRIST)
    r_sh, r_el, r_wr = _get(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER), _get(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW), _get(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST)
    la = _angle_3d(l_sh, l_el, l_wr)
    ra = _angle_3d(r_sh, r_el, r_wr)
    return None if la is None or ra is None else min(la, ra)

def _squat_angle(landmarks):
    l_hip, l_knee, l_ank = _get(landmarks, mp_pose.PoseLandmark.LEFT_HIP), _get(landmarks, mp_pose.PoseLandmark.LEFT_KNEE), _get(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
    r_hip, r_knee, r_ank = _get(landmarks, mp_pose.PoseLandmark.RIGHT_HIP), _get(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE), _get(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE)
    la = _angle_3d(l_hip, l_knee, l_ank)
    ra = _angle_3d(r_hip, r_knee, r_ank)
    return None if la is None or ra is None else min(la, ra)


# ======================
# 3) 실제 분석 함수 (모델은 외부에서 주입받음)
# ======================
def analyze_video_run(video_path: str, exercise: str, pose_model) -> dict:
    exercise = exercise.lower().strip()
    if exercise not in {"pushup", "squat"}:
        return {"error": f"Unsupported exercise: {exercise}. Use 'pushup' or 'squat'."}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Failed to open video."}

    if exercise == "pushup":
        angle_fn = _pushup_angle
        down_thr, up_thr, strict_min, min_duration_s = 90, 150, 80, 0.3
        exercise_type = "pushup"
    else:
        angle_fn = _squat_angle
        down_thr, up_thr, strict_min, min_duration_s = 80, 165, 70, 0.35
        exercise_type = "squat"

    state, rep_count, count_incorrect = "up", 0, 0
    cur_rep_min_angle, rep_start_time = 999.0, None
    too_shallow, too_fast, unstable = 0, 0, 0

    t0 = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = pose_model.process(img_rgb)

        if not results.pose_landmarks:
            unstable += 1
            continue

        ang = angle_fn(results.pose_landmarks.landmark)
        if ang is None:
            unstable += 1
            continue

        if state == "up":
            if ang < down_thr:
                state = "down"
                rep_start_time = time.perf_counter()
                cur_rep_min_angle = ang
        else:  # down
            cur_rep_min_angle = min(cur_rep_min_angle, ang)
            if ang > up_thr:
                state = "up"
                rep_count += 1
                poor = False
                if cur_rep_min_angle > strict_min:
                    too_shallow += 1
                    poor = True
                if rep_start_time and (time.perf_counter() - rep_start_time) < min_duration_s:
                    too_fast += 1
                    poor = True
                if poor:
                    count_incorrect += 1

    cap.release()
    elapsed_time = float(time.perf_counter() - t0)

    msgs = []
    if too_shallow > 0:
        msgs.append("동작 깊이가 부족했습니다.")
    if too_fast > 0:
        msgs.append("동작 속도가 너무 빨랐습니다.")
    if unstable > 0:
        msgs.append("자세 인식이 불안정했습니다. 카메라 각도/조명을 조정해보세요.")
    if not msgs and rep_count > 0:
        msgs.append("좋아요! 전반적으로 안정적인 폼입니다.")

    return {
        "exercise_type": exercise_type,
        "rep_count": rep_count,
        "avg_accuracy": rep_count,
    }
