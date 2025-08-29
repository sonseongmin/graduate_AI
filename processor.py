# processor.py
import cv2
import numpy as np
import mediapipe as mp
import time

mp_pose = mp.solutions.pose

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
    if la is None or ra is None:
        return None
    return min(la, ra)

def _squat_angle(landmarks):
    l_hip, l_knee, l_ank = _get(landmarks, mp_pose.PoseLandmark.LEFT_HIP), _get(landmarks, mp_pose.PoseLandmark.LEFT_KNEE), _get(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
    r_hip, r_knee, r_ank = _get(landmarks, mp_pose.PoseLandmark.RIGHT_HIP), _get(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE), _get(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE)
    la = _angle_3d(l_hip, r_knee, r_ank) if False else _angle_3d(l_hip, l_knee, l_ank)  # left
    ra = _angle_3d(r_hip, r_knee, r_ank)  # right
    if la is None or ra is None:
        return None
    return min(la, ra)

def analyze_video(video_path: str, exercise: str):
    """분석 결과를 지정한 스키마로 반환"""
    exercise = exercise.lower().strip()
    if exercise not in {"pushup", "squat"}:
        return {"error": f"Unsupported exercise: {exercise}. Use 'pushup' or 'squat'."}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Failed to open video."}

    # 임계값(필요 시 조정)
    if exercise == "pushup":
        angle_fn = _pushup_angle
        down_thr = 90
        up_thr   = 150
        strict_min = 80
        min_duration_s = 0.3
        exercise_name = "pushup"
    else:
        angle_fn = _squat_angle
        down_thr = 80
        up_thr   = 165
        strict_min = 70
        min_duration_s = 0.35
        exercise_name = "squat"

    state = "up"           # up -> down -> up을 1회로 카운트
    count_total = 0
    count_incorrect = 0

    cur_rep_min_angle = 999.0
    rep_start_time = None
    too_shallow = 0
    too_fast = 0
    unstable = 0

    t0 = time.perf_counter()

    with mp_pose.Pose(static_image_mode=False, model_complexity=1) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_rgb.flags.writeable = False
            results = pose.process(img_rgb)

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
                    count_total += 1
                    poor = False
                    if cur_rep_min_angle > strict_min:
                        too_shallow += 1
                        poor = True
                    if rep_start_time is not None and (time.perf_counter() - rep_start_time) < min_duration_s:
                        too_fast += 1
                        poor = True
                    if poor:
                        count_incorrect += 1

    cap.release()
    elapsed_time = float(time.perf_counter() - t0)

    # 단일 문자열 피드백 생성
    msgs = []
    if too_shallow > 0:
        msgs.append("동작 깊이가 부족했습니다.")
    if too_fast > 0:
        msgs.append("동작 속도가 너무 빨랐습니다.")
    if unstable > 0:
        msgs.append("자세 인식이 불안정한 프레임이 있었습니다. 카메라 각도/조명을 조정해보세요.")
    if not msgs and count_total > 0:
        msgs.append("좋아요! 전반적으로 안정적인 폼입니다.")
    feedback = msgs

    return {
        "exercise_name": exercise_name,
        "count_total": int(count_total),
        "count_incorrect": int(count_incorrect),
        "feedback": feedback,
        "elapsed_time": round(elapsed_time, 3)
    }
