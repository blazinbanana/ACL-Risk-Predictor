import os
import cv2
import numpy as np
from collections import deque
import mediapipe as mp
from tensorflow.keras.models import load_model
from pose_utils import extract_frame_features

# --- Detect if running in Hugging Face Spaces ---
RUNNING_IN_HF = os.getenv("SYSTEM") == "spaces"
OUTPUT_DIR = "/tmp" if RUNNING_IN_HF else "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def safe_video_writer(output_path, width, height, fps):
    """
    Try 'avc1' (browser-friendly H.264) first,
    fallback to 'mp4v' if H.264 isn’t supported.
    """
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            raise ValueError("avc1 not supported")
        print("[INFO] Using H.264 (avc1) codec")
        return out
    except Exception:
        print("[WARN] Falling back to mp4v codec")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def analyze_video_with_overlay(video_path, model, output_path=None):
    """
    Run trained ACL risk model on a video and overlay prediction feedback.
    """
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "output_video.mp4")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {video_path}")

    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = safe_video_writer(output_path, width, height, fps)

    frame_buffer = deque(maxlen=30)
    risk_level, confidence = None, None
    written_frames = 0

    print(f"[INFO] Starting analysis for {video_path}")
    print(f"[INFO] Output video: {output_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            features = extract_frame_features(results.pose_landmarks)
            frame_buffer.append(features)

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
        written_frames += 1

    cap.release()
    out.release()

    print(f"[INFO] Total frames written: {written_frames}")

    if written_frames == 0 or not os.path.exists(output_path):
        raise FileNotFoundError(f"Processed video not found at {output_path} — analysis may have failed.")
        
    # --- Convert for browser playback ---
    import subprocess
    try:
        h264_output = output_path.replace(".mp4", "_web.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-i", output_path,
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            h264_output
        ]
        subprocess.run(cmd, check=True)
        print(f"[SUCCESS] Converted to browser-playable format: {h264_output}")
        output_path = h264_output
    except Exception as e:
        print(f"[WARN] FFmpeg conversion failed: {e}")

    print(f"[SUCCESS] Analysis complete. Video saved to {output_path}")
    return output_path, risk_level, confidence


if __name__ == "__main__":
    model = load_model("model/acl_risk_model_v2.h5")
    analyze_video_with_overlay("acl_data/P01_front_high.mp4", model)
