import cv2
import mediapipe as mp
import os
import json
import numpy as np
from pathlib import Path
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress TensorFlow and C++ warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)
warnings.filterwarnings("ignore")
sys.stderr = open(os.devnull, 'w')

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

DATA_DIR = Path(r'f:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\data\processed')
OUTPUT_DIR = Path(r'f:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\data\keypoints')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def standardize_sequence_length(keypoints, target_length=16):
    if len(keypoints) == 1:
        return None
    if len(keypoints) > target_length:
        indices = np.linspace(0, len(keypoints) - 1, target_length, dtype=int)
        keypoints = [keypoints[i] for i in indices]
    elif len(keypoints) < target_length:
        keypoints += [keypoints[-1]] * (target_length - len(keypoints))
    return keypoints

def process_instance(instance_dir, label):
    images = sorted(instance_dir.glob('*.jpg'))
    if len(images) <= 1:
        print(f"Skipping {instance_dir} due to insufficient frames.")
        return

    hands_detector = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
    pose_detector = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    keypoints = []
    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_results = hands_detector.process(frame_rgb)
        pose_results = pose_detector.process(frame_rgb)

        frame_keypoints = {}
        if hand_results.multi_hand_landmarks:
            frame_keypoints['hands'] = []
            for hand_landmarks in hand_results.multi_hand_landmarks:
                hand_points = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                frame_keypoints['hands'].append(hand_points)

        if pose_results.pose_landmarks:
            frame_keypoints['pose'] = [[lm.x, lm.y, lm.z] for lm in pose_results.pose_landmarks.landmark]

        keypoints.append(frame_keypoints)

    hands_detector.close()
    pose_detector.close()

    keypoints = standardize_sequence_length(keypoints)
    if keypoints is None:
        print(f"Skipping {instance_dir} after standardization.")
        return

    output_path = OUTPUT_DIR / label / f"{instance_dir.name}_keypoints.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({
            "label": label,
            "instance": instance_dir.name,
            "keypoints": keypoints
        }, f)

    print(f"Saved {output_path}")

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
