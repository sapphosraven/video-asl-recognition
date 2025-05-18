import cv2
import mediapipe as mp
import os
import json
import numpy as np
from pathlib import Path
import sys
import logging
import warnings
from tqdm import tqdm

# Suppress TensorFlow and C++ warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Suppress absl logging from MediaPipe
os.environ['GLOG_minloglevel'] = '2'
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)

# Suppress all warnings
warnings.filterwarnings("ignore")

# Redirect STDERR to suppress remaining warnings
sys.stderr = open(os.devnull, 'w')

# Initialize MediaPipe Hands and Pose modules
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# Configuration
DATA_DIR = Path(r'f:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\data\processed')  # use processed frames per label/instance
OUTPUT_DIR = Path(r'f:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\data\keypoints')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Function to standardize sequence length
def standardize_sequence_length(keypoints, target_length=60):
    if len(keypoints) == 1:
        # Ignore videos with only 1 frame (buggy videos)
        return None

    if len(keypoints) > target_length:
        # Uniformly sample frames if the sequence is too long
        indices = np.linspace(0, len(keypoints) - 1, target_length, dtype=int)
        keypoints = [keypoints[i] for i in indices]
    elif len(keypoints) < target_length:
        # Pad with the last frame if the sequence is too short
        keypoints += [keypoints[-1]] * (target_length - len(keypoints))

    return keypoints

# Function to extract label and instance number from the directory structure
def extract_label_and_instance(video_path):
    label = video_path.parent.name  # Parent directory name is the label
    instance = video_path.stem  # Video file name (without extension) is the instance number
    return label, instance

# Function to process a single video and extract keypoints
def process_video(video_path, output_path):
    cap = cv2.VideoCapture(str(video_path))
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    keypoints = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe Hands
        hand_results = hands.process(frame_rgb)
        pose_results = pose.process(frame_rgb)

        frame_keypoints = {}

        # Extract hand keypoints
        if hand_results.multi_hand_landmarks:
            frame_keypoints['hands'] = []
            for hand_landmarks in hand_results.multi_hand_landmarks:
                hand_points = []
                for landmark in hand_landmarks.landmark:
                    hand_points.append((landmark.x, landmark.y, landmark.z))
                frame_keypoints['hands'].append(hand_points)

        # Extract pose keypoints
        if pose_results.pose_landmarks:
            frame_keypoints['pose'] = []
            for landmark in pose_results.pose_landmarks.landmark:
                frame_keypoints['pose'].append((landmark.x, landmark.y, landmark.z))

        keypoints.append(frame_keypoints)

    cap.release()
    hands.close()
    pose.close()

    # Standardize sequence length
    keypoints = standardize_sequence_length(keypoints)
    if keypoints is None:
        print(f"Skipping {video_path} due to insufficient frames.")
        return

    # Extract label and instance number
    label, instance = extract_label_and_instance(video_path)

    # Save keypoints to JSON with label and instance
    output_data = {
        "label": label,
        "instance": instance,
        "keypoints": keypoints
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f)

# Process keypoint extraction on frame directories
for label_dir in tqdm(DATA_DIR.iterdir(), desc="Processing labels"):
    if not label_dir.is_dir():
        continue
    for instance_dir in tqdm(label_dir.iterdir(), desc=f"Processing instances in {label_dir.name}", leave=False):
        if not instance_dir.is_dir():
            continue
        images = sorted(instance_dir.glob('*.jpg'))
        if len(images) <= 1:
            print(f"Skipping {instance_dir} due to insufficient frames.")
            continue
        keypoints = []
        # Initialize detectors once per instance
        hands_detector = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
        pose_detector = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        for img_path in tqdm(images, desc=f"Processing frames in {instance_dir.name}", leave=False):
            frame = cv2.imread(str(img_path))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands_detector.process(frame_rgb)
            pose_results = pose_detector.process(frame_rgb)
            frame_keypoints = {}
            # Extract hand keypoints
            if hand_results.multi_hand_landmarks:
                frame_keypoints['hands'] = []
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    hand_points = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                    frame_keypoints['hands'].append(hand_points)
            # Extract pose keypoints
            if pose_results.pose_landmarks:
                frame_keypoints['pose'] = [(lm.x, lm.y, lm.z) for lm in pose_results.pose_landmarks.landmark]
            keypoints.append(frame_keypoints)
        # Close detectors
        hands_detector.close()
        pose_detector.close()
        # Standardize and save
        keypoints = standardize_sequence_length(keypoints)
        if keypoints is None:
            print(f"Skipping {instance_dir} after standardization.")
            continue
        label = label_dir.name
        instance = instance_dir.name
        output_data = {"label": label, "instance": instance, "keypoints": keypoints}
        output_file = OUTPUT_DIR / label / f"{instance}_keypoints.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving {output_file}")
        with open(output_file, 'w') as f:
            json.dump(output_data, f)
