# video_analysis.py

import os
import cv2
import numpy as np
from collections import deque
import mediapipe as mp
from tensorflow.keras.models import load_model
from pose_utils import extract_frame_features, calculate_biomechanical_features

# --- Ensure output directory exists ---
os.makedirs("output", exist_ok=True)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def analyze_video_with_overlay(video_path, model, output_path='output/output_video.mp4'):
    """
    Run trained ACL risk model on a video and overlay prediction feedback
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    # Define output video writer
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        cap.get(cv2.CAP_PROP_FPS) or 30.0,
        (int(cap.get(3)), int(cap.get(4)))
    )

    frame_buffer = deque(maxlen=30)
    risk_level, confidence = None, None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            features = extract_frame_features(results.pose_landmarks)
            frame_buffer.append(features)

            # Once 30 frames are collected, make prediction
            if len(frame_buffer) == 30:
                sequence = np.array(frame_buffer).reshape(1, 30, -1)
                prediction = model.predict(sequence, verbose=0)
                risk_level = np.argmax(prediction, axis=1)[0]
                confidence = np.max(prediction)

                risk_text = "HIGH RISK" if risk_level == 1 else "LOW RISK"
                color = (0, 0, 255) if risk_level == 1 else (0, 255, 0)

                cv2.putText(frame, f"ACL RISK: {risk_text}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                if risk_level == 1:
                    cv2.putText(frame, "Issue: Knee Valgus Detected", (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        out.write(frame)

    cap.release()
    out.release()

    if not os.path.exists(output_path):
        raise FileNotFoundError("Processed video not found — analysis may have failed.")

    print(f"Analysis complete. Output saved to: {output_path}")
    return output_path, risk_level, confidence


if __name__ == "__main__":
    model = load_model("model/acl_risk_model_v2.h5")
    analyze_video_with_overlay("acl_data/P01_front_high.mp4", model)
