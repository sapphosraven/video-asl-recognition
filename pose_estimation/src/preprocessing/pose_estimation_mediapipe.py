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

    hands_detector = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
    pose_detector = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    face_detector = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

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
        
        # Fill hand landmarks (33-74)
        if hand_results.multi_hand_landmarks:
            for h_i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                offset = NUM_POSE + h_i * NUM_HAND
                for j, lm in enumerate(hand_landmarks.landmark):
                    nodes[offset + j] = [lm.x, lm.y, lm.z]
        
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

    # save full sequence in compressed NPZ instead of JSON
    output_path = OUTPUT_DIR / label / f"{instance_dir.name}_keypoints.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output_path),
        nodes=np.array(all_nodes, dtype=np.float32),
        label=label,
        instance=instance_dir.name
    )
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
