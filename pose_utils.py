import cv2
import numpy as np
import mediapipe as mp
# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
def extract_pose_sequences(video_path, sequence_length=30):
    """
    Extract pose sequences from video using MediaPipe
    Returns: Normalized pose sequences and confidence scores
    """
    cap = cv2.VideoCapture(video_path)
    sequences = []
    confidence_scores = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            # Extract keypoints (we'll use hips, knees, ankles, shoulders)
            landmarks = results.pose_landmarks.landmark
            
            # Define key joints for ACL risk analysis
            key_joints = {
                'left_hip': [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
                'right_hip': [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
                'left_knee': [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
                'right_knee': [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y],
                'left_ankle': [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y],
                'right_ankle': [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y],
                'left_shoulder': [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
                'right_shoulder': [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            }
            
            # Convert to array and normalize relative to hips
            pose_data = []
            for joint in ['left_hip', 'right_hip', 'left_knee', 'right_knee', 
                         'left_ankle', 'right_ankle', 'left_shoulder', 'right_shoulder']:
                # Normalize coordinates relative to midpoint between hips
                hip_mid_x = (key_joints['left_hip'][0] + key_joints['right_hip'][0]) / 2
                hip_mid_y = (key_joints['left_hip'][1] + key_joints['right_hip'][1]) / 2
                
                norm_x = key_joints[joint][0] - hip_mid_x
                norm_y = key_joints[joint][1] - hip_mid_y
                
                pose_data.extend([norm_x, norm_y])
            
            sequences.append(pose_data)
            confidence_scores.append(1.0)  # High confidence when landmarks detected
        else:
            # If no landmarks detected, use zeros with low confidence
            sequences.append([0] * 16)  # 8 joints * 2 coordinates
            confidence_scores.append(0.0)
    
    cap.release()
    
    # Convert to numpy arrays
    sequences = np.array(sequences)
    confidence_scores = np.array(confidence_scores)
    
    return sequences, confidence_scores


def calculate_biomechanical_features(pose_sequence):
    """
    Calculate specific biomechanical features from pose sequences
    """
    features = []
    
    for i in range(len(pose_sequence)):
        # Extract coordinates (assuming the sequence structure from previous function)
        # Indices: 0-1: left_hip, 2-3: right_hip, 4-5: left_knee, 6-7: right_knee, etc.
        left_knee_x, left_knee_y = pose_sequence[i][4], pose_sequence[i][5]
        right_knee_x, right_knee_y = pose_sequence[i][6], pose_sequence[i][7]
        left_ankle_x, left_ankle_y = pose_sequence[i][8], pose_sequence[i][9]
        right_ankle_x, right_ankle_y = pose_sequence[i][10], pose_sequence[i][11]
        left_shoulder_x, left_shoulder_y = pose_sequence[i][12], pose_sequence[i][13]
        right_shoulder_x, right_shoulder_y = pose_sequence[i][14], pose_sequence[i][15]
        
        # 1. Knee Valgus Angle (simplified 2D calculation)
        # For left leg: angle between hip-knee and knee-ankle vectors
        left_hip_knee_vec = np.array([left_knee_x, left_knee_y])
        left_knee_ankle_vec = np.array([left_ankle_x - left_knee_x, 
                                      left_ankle_y - left_knee_y])
        
        if np.linalg.norm(left_hip_knee_vec) > 0 and np.linalg.norm(left_knee_ankle_vec) > 0:
            left_valgus_angle = np.degrees(np.arccos(
                np.dot(left_hip_knee_vec, left_knee_ankle_vec) / 
                (np.linalg.norm(left_hip_knee_vec) * np.linalg.norm(left_knee_ankle_vec))
            ))
        else:
            left_valgus_angle = 0
        
        # 2. Knee Flexion (simplified)
        left_flexion = np.degrees(np.arctan2(abs(left_ankle_y - left_knee_y), 
                                           abs(left_ankle_x - left_knee_x)))
        
        # 3. Trunk Lean
        shoulder_mid_x = (left_shoulder_x + right_shoulder_x) / 2
        hip_mid_x = (pose_sequence[i][0] + pose_sequence[i][2]) / 2
        trunk_lean = abs(shoulder_mid_x - hip_mid_x)
        
        features.append([left_valgus_angle, left_flexion, trunk_lean])
    
    return np.array(features)


def prepare_dataset(video_folder, labels, sequence_length=30):
    """
    Prepare dataset from video folder
    video_folder: dictionary with {'video_path': 'label'}
    labels: dictionary mapping label names to integers
    """
    X_sequences = []
    y_labels = []
    
    for video_path, label in video_folder.items():
        print(f"Processing {video_path}...")
        
        # Extract pose sequences
        pose_seq, confidence = extract_pose_sequences(video_path, sequence_length)
        
        # Calculate biomechanical features
        bio_features = calculate_biomechanical_features(pose_seq)
        
        # Create sequences for LSTM
        for i in range(len(bio_features) - sequence_length + 1):
            sequence = bio_features[i:i + sequence_length]
            X_sequences.append(sequence)
            y_labels.append(labels[label])
    
    return np.array(X_sequences), np.array(y_labels)



def extract_frame_features(landmarks):
    """
    Extract biomechanical features for a single frame (1D array of 3 features)
    Reuses logic similar to calculate_biomechanical_features()
    """
    # Define key joints
    def get_xy(j):
        return [landmarks.landmark[j].x, landmarks.landmark[j].y]

    key_joints = {
        'left_hip': get_xy(mp_pose.PoseLandmark.LEFT_HIP.value),
        'right_hip': get_xy(mp_pose.PoseLandmark.RIGHT_HIP.value),
        'left_knee': get_xy(mp_pose.PoseLandmark.LEFT_KNEE.value),
        'right_knee': get_xy(mp_pose.PoseLandmark.RIGHT_KNEE.value),
        'left_ankle': get_xy(mp_pose.PoseLandmark.LEFT_ANKLE.value),
        'right_ankle': get_xy(mp_pose.PoseLandmark.RIGHT_ANKLE.value),
        'left_shoulder': get_xy(mp_pose.PoseLandmark.LEFT_SHOULDER.value),
        'right_shoulder': get_xy(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
    }

    # Normalize relative to hip midpoint
    hip_mid_x = (key_joints['left_hip'][0] + key_joints['right_hip'][0]) / 2
    hip_mid_y = (key_joints['left_hip'][1] + key_joints['right_hip'][1]) / 2
    for j in key_joints:
        key_joints[j][0] -= hip_mid_x
        key_joints[j][1] -= hip_mid_y

    # Compute features (same as calculate_biomechanical_features)
    lkx, lky = key_joints['left_knee']
    lax, lay = key_joints['left_ankle']
    lhx, lhy = key_joints['left_hip']
    lsx, lsy = key_joints['left_shoulder']
    rsx, rsy = key_joints['right_shoulder']

    # Knee valgus angle
    hip_knee = np.array([lhx - lkx, lhy - lky])
    knee_ankle = np.array([lax - lkx, lay - lky])
    if np.linalg.norm(hip_knee) > 0 and np.linalg.norm(knee_ankle) > 0:
        valgus = np.degrees(np.arccos(
            np.dot(hip_knee, knee_ankle) /
            (np.linalg.norm(hip_knee) * np.linalg.norm(knee_ankle))
        ))
    else:
        valgus = 0

    # Knee flexion
    flexion = np.degrees(np.arctan2(abs(lay - lky), abs(lax - lkx)))

    # Trunk lean
    shoulder_mid_x = (lsx + rsx) / 2
    trunk_lean = abs(shoulder_mid_x - hip_mid_x)

    return np.array([valgus, flexion, trunk_lean])