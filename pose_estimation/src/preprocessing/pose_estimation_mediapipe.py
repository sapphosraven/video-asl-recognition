import cv2
import mediapipe as mp
import os
import json
import numpy as np
from pathlib import Path
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.interpolate import interp1d

# Suppress TensorFlow and C++ warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)
warnings.filterwarnings("ignore")
sys.stderr = open(os.devnull, 'w')

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

DATA_DIR = Path(r'f:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\data\processed')
OUTPUT_DIR = Path(r'f:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\data\keypoints')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_POSE = 33
NUM_HAND = 21
NUM_FACE = 478  # MediaPipe face mesh provides 478 landmarks
TOTAL_NODES = NUM_POSE + 2 * NUM_HAND + NUM_FACE  # 33 + 42 + 478 = 553

def process_instance(instance_dir, label):
    images = sorted(instance_dir.glob('*.jpg'))
    if len(images) <= 1:
        print(f"Skipping {instance_dir} due to insufficient frames.")
        return

    # OPTIMIZED SETTINGS FOR ASL HAND DETECTION
    hands_detector = mp_hands.Hands(
        static_image_mode=True, 
        max_num_hands=2,                    # Always expect 2 hands in ASL
        min_detection_confidence=0.2,       # MUCH LOWER for difficult ASL videos
        min_tracking_confidence=0.2,        # MUCH LOWER 
        model_complexity=1                  # Higher complexity model
    )
    pose_detector = mp_pose.Pose(
        static_image_mode=True, 
        model_complexity=2,                 # Highest complexity
        min_detection_confidence=0.3        # Lower threshold
    )
    face_detector = mp_face_mesh.FaceMesh(
        static_image_mode=True, 
        max_num_faces=1, 
        refine_landmarks=True, 
        min_detection_confidence=0.3        # Lower threshold
    )

    all_nodes = []
    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_results = hands_detector.process(frame_rgb)
        pose_results = pose_detector.process(frame_rgb)
        face_results = face_detector.process(frame_rgb)        
        nodes = np.zeros((TOTAL_NODES, 3), dtype=np.float32)
        
        # Fill pose landmarks (0-32)
        if pose_results.pose_landmarks:
            for i, lm in enumerate(pose_results.pose_landmarks.landmark):
                nodes[i] = [lm.x, lm.y, lm.z]
        
        # Fill hand landmarks (33-74) with improved handling
        hands_filled = [False, False]  # Track which hands were detected
        if hand_results.multi_hand_landmarks:
            for h_i, (hand_landmarks, handedness) in enumerate(zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness)):
                # Determine if this is left or right hand
                hand_label = handedness.classification[0].label
                hand_idx = 0 if hand_label == 'Left' else 1  # MediaPipe perspective
                
                offset = NUM_POSE + hand_idx * NUM_HAND
                for j, lm in enumerate(hand_landmarks.landmark):
                    nodes[offset + j] = [lm.x, lm.y, lm.z]
                hands_filled[hand_idx] = True
        
        # CRITICAL: Mark frames where hands were detected for later interpolation
        hand_detection_status = {
            'left_hand': hands_filled[0],
            'right_hand': hands_filled[1],
            'frame_idx': len(all_nodes)
        }
        
        # Fill face landmarks (75-542)
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                offset = NUM_POSE + 2 * NUM_HAND  # 75
                for i, lm in enumerate(face_landmarks.landmark):
                    nodes[offset + i] = [lm.x, lm.y, lm.z]        
        all_nodes.append(nodes.tolist())

    hands_detector.close()
    pose_detector.close()
    face_detector.close()

    # CRITICAL: Apply hand landmark interpolation for missing detections
    all_nodes = interpolate_missing_hands(all_nodes)
    
    # Calculate detection statistics
    total_frames = len(all_nodes)
    hand_detection_count = sum(1 for frame in all_nodes if np.any(frame[NUM_POSE:NUM_POSE+2*NUM_HAND] != 0))
    hand_detection_rate = (hand_detection_count / total_frames) * 100 if total_frames > 0 else 0
    
    print(f"  Hand detection: {hand_detection_rate:.1f}% ({hand_detection_count}/{total_frames} frames)")

    # save full sequence in compressed NPZ instead of JSON
    output_path = OUTPUT_DIR / label / f"{instance_dir.name}_keypoints.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output_path),
        nodes=np.array(all_nodes, dtype=np.float32),
        label=label,
        instance=instance_dir.name,
        hand_detection_rate=hand_detection_rate  # Store detection rate for analysis
    )
    print(f"Saved {output_path}")

def interpolate_missing_hands(all_nodes):
    """Interpolate missing hand landmarks using pose information and temporal smoothing"""
    if len(all_nodes) <= 1:
        return all_nodes
    
    all_nodes = np.array(all_nodes)
    seq_len, num_nodes, coords = all_nodes.shape
    
    # Define hand ranges
    left_hand_range = range(NUM_POSE, NUM_POSE + NUM_HAND)  # 33-53
    right_hand_range = range(NUM_POSE + NUM_HAND, NUM_POSE + 2*NUM_HAND)  # 54-74
    
    # Interpolate each hand separately
    for hand_range, hand_name in [(left_hand_range, "left"), (right_hand_range, "right")]:
        # Find frames where this hand is detected (non-zero)
        detected_frames = []
        for frame_idx in range(seq_len):
            if np.any(all_nodes[frame_idx, hand_range, :] != 0):
                detected_frames.append(frame_idx)
        
        if len(detected_frames) >= 2:
            # Linear interpolation between detected frames
            from scipy.interpolate import interp1d
            
            for coord_idx in range(coords):
                for landmark_idx in hand_range:
                    # Get detected values
                    detected_values = all_nodes[detected_frames, landmark_idx, coord_idx]
                    
                    # Create interpolation function
                    f = interp1d(detected_frames, detected_values, 
                               kind='linear', fill_value='extrapolate', bounds_error=False)
                    
                    # Interpolate missing frames
                    all_frame_indices = np.arange(seq_len)
                    interpolated_values = f(all_frame_indices)
                    
                    # Only fill missing frames (keep original detections)
                    for frame_idx in range(seq_len):
                        if frame_idx not in detected_frames:
                            all_nodes[frame_idx, landmark_idx, coord_idx] = interpolated_values[frame_idx]
        
        elif len(detected_frames) == 1:
            # Forward/backward fill for single detection
            single_frame = detected_frames[0]
            single_hand_data = all_nodes[single_frame, hand_range, :]
            
            # Fill all frames with this detection
            for frame_idx in range(seq_len):
                if np.all(all_nodes[frame_idx, hand_range, :] == 0):
                    all_nodes[frame_idx, hand_range, :] = single_hand_data
    
    return all_nodes.tolist()

def main():
    label_dirs = [d for d in DATA_DIR.iterdir() if d.is_dir()]
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        for label_dir in label_dirs:
            label = label_dir.name
            for instance_dir in label_dir.iterdir():
                if not instance_dir.is_dir():
                    continue
                futures.append(executor.submit(process_instance, instance_dir, label))

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing instance: {e}")

if __name__ == "__main__":
    main()
