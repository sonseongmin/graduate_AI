import cv2
import mediapipe as mp
import numpy as np
from collections import deque

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle_3d(a, b, c):
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def main():
    video_path = 'push_up_1.mp4'  # 비디오 경로
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("비디오 열기 실패")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    initial_skip_sec = 2      # 초기 무시 시간 2초로 변경
    classify_sec = 2          # 분류에 사용할 프레임 길이 2초
    initial_skip_frames = int(fps * initial_skip_sec)
    classify_frames = int(fps * classify_sec)

    count = 0
    stage = None
    motion_type = None

    elbow_angles = deque(maxlen=classify_frames)
    knee_angles = deque(maxlen=classify_frames)

    frame_count = 0

    with mp_pose.Pose(model_complexity=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:

        # --- 1차 탐색 : 초기 2초 프레임 무시 + 2초 데이터로 운동 종류 판단 ---
        while motion_type is None:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            if frame_count < initial_skip_frames:
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # 팔꿈치 각도 계산
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

                left_elbow_angle = calculate_angle_3d(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = calculate_angle_3d(right_shoulder, right_elbow, right_wrist)
                avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
                elbow_angles.append(avg_elbow_angle)

                # 무릎 각도 계산
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
                left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
                right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

                left_knee_angle = calculate_angle_3d(left_hip, left_knee, left_ankle)
                right_knee_angle = calculate_angle_3d(right_hip, right_knee, right_ankle)
                avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
                knee_angles.append(avg_knee_angle)

                if len(elbow_angles) == classify_frames and len(knee_angles) == classify_frames:
                    elbow_var = max(elbow_angles) - min(elbow_angles)
                    knee_var = max(knee_angles) - min(knee_angles)
                    elbow_mean = sum(elbow_angles) / len(elbow_angles)
                    knee_mean = sum(knee_angles) / len(knee_angles)

                    # 푸쉬업 조건: 팔꿈치 움직임 폭 크고, 평균 각도 범위 (50~160도 정도)
                    pushup_cond = (elbow_var > knee_var) and (50 < elbow_mean < 160)
                    # 스쿼트 조건: 무릎 움직임 폭 크고, 평균 각도 범위 (50~160도 정도)
                    squat_cond = (knee_var >= elbow_var) and (50 < knee_mean < 160)

                    if pushup_cond:
                        motion_type = "pushup"
                        print("운동 감지: 푸시업")
                    elif squat_cond:
                        motion_type = "squat"
                        print("운동 감지: 스쿼트")
                    else:
                        print("운동 분류 불가, 데이터 더 필요함")
                        # 초기 프레임 더 읽어도 되고, 반복하거나 종료 가능
                        # 여기선 계속 탐색 유지

        cap.release()

        # --- 2차 재생 : 처음부터 재생하며 카운팅 ---
        cap = cv2.VideoCapture(video_path)
        count = 0
        stage = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # 팔꿈치 각도
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

                left_elbow_angle = calculate_angle_3d(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = calculate_angle_3d(right_shoulder, right_elbow, right_wrist)
                avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2

                # 무릎 각도
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
                left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
                right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

                left_knee_angle = calculate_angle_3d(left_hip, left_knee, left_ankle)
                right_knee_angle = calculate_angle_3d(right_hip, right_knee, right_ankle)
                avg_knee_angle = (left_knee_angle + right_knee_angle) / 2

                if motion_type == "pushup":
                    DOWN_ANGLE = 100
                    UP_ANGLE = 150
                    if avg_elbow_angle < DOWN_ANGLE:
                        stage = 'down'
                    if avg_elbow_angle > UP_ANGLE and stage == 'down':
                        stage = 'up'
                        count += 1
                        print(f"푸시업 횟수: {count}")

                    display_text = f"Pushup Count: {count}"
                    angle_text = f"Elbow Angle: {int(avg_elbow_angle)}"

                elif motion_type == "squat":
                    DOWN_ANGLE = 90
                    UP_ANGLE = 120
                    if avg_knee_angle < DOWN_ANGLE:
                        stage = 'down'
                    if avg_knee_angle > UP_ANGLE and stage == 'down':
                        stage = 'up'
                        count += 1
                        print(f"스쿼트 횟수: {count}")

                    display_text = f"Squat Count: {count}"
                    angle_text = f"Knee Angle: {int(avg_knee_angle)}"

                else:
                    display_text = "운동 감지 중..."
                    angle_text = ""

                cv2.putText(image, display_text, (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                if angle_text:
                    cv2.putText(image, angle_text, (30, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            else:
                cv2.putText(image, "포즈 인식 불가", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            display_img = cv2.resize(image, (1000, 800))
            cv2.imshow('운동 카운터', display_img)

            key = cv2.waitKey(10) & 0xFF
            if key == 27:
                print("사용자 종료")
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
