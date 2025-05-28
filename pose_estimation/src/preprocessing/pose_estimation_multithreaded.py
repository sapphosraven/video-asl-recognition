"""
Multi-threaded 553-node keypoint extraction for ASL recognition.
Uses conservative 4-worker threading for 4x speedup while maintaining stability.
"""

import cv2
import mediapipe as mp
import os
import numpy as np
from pathlib import Path
import warnings
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import time

# Suppress TensorFlow and C++ warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

DATA_DIR = Path(r'f:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\data\processed')
OUTPUT_DIR = Path(r'f:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\data\keypoints')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_POSE = 33
NUM_HAND = 21
NUM_FACE = 478  # MediaPipe face mesh provides 478 landmarks
TOTAL_NODES = NUM_POSE + 2 * NUM_HAND + NUM_FACE  # 553

# Thread-safe progress tracking
progress_lock = threading.Lock()
progress_stats = {
    'processed': 0,
    'failed': 0,
    'total': 0,
    'start_time': time.time()
}

def print_progress():
    """Print thread-safe progress updates"""
    with progress_lock:
        elapsed = time.time() - progress_stats['start_time']
        rate = progress_stats['processed'] / elapsed if elapsed > 0 else 0
        remaining = progress_stats['total'] - progress_stats['processed'] - progress_stats['failed']
        eta = remaining / rate if rate > 0 else 0
        
        print(f"\n\n\n\n📊 Progress: {progress_stats['processed']}/{progress_stats['total']}"
              f"({progress_stats['processed']/progress_stats['total']*100:.1f}%) "
              f"| Failed: {progress_stats['failed']} "
              f"| Rate: {rate:.1f}/min "
              f"| ETA: {eta/60:.1f}min\n\n\n\n")

def process_single_instance(instance_info):
    """Process a single video instance with memory efficiency"""
    instance_dir, label = instance_info
    
    images = sorted(instance_dir.glob('*.jpg'))
    if len(images) <= 1:
        with progress_lock:
            progress_stats['failed'] += 1
        return False, f"Insufficient frames ({len(images)})"
    
    # Check if already processed
    output_path = OUTPUT_DIR / label / f"{instance_dir.name}_keypoints.npz"
    if output_path.exists():
        with progress_lock:
            progress_stats['processed'] += 1
        return True, "Already exists"
    
    # Initialize MediaPipe detectors (thread-local)
    hands_detector = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.2,
        min_tracking_confidence=0.2
    )
    
    pose_detector = mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.3
    )
    
    face_detector = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3
    )
    
    all_keypoints = []
    hand_detection_count = 0
    
    try:
        for frame_idx, img_path in enumerate(images):
            # Load and process single frame
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run MediaPipe detection
            hand_results = hands_detector.process(frame_rgb)
            pose_results = pose_detector.process(frame_rgb)
            face_results = face_detector.process(frame_rgb)
            
            # Create keypoint array for this frame
            keypoints = np.zeros((TOTAL_NODES, 3), dtype=np.float32)
            
            # Fill pose landmarks (0-32)
            if pose_results.pose_landmarks:
                for i, lm in enumerate(pose_results.pose_landmarks.landmark):
                    keypoints[i] = [lm.x, lm.y, lm.z]
            
            # Fill hand landmarks (33-74)
            if hand_results.multi_hand_landmarks:
                hand_detection_count += 1
                for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    if hand_idx < 2:  # Only process first 2 hands
                        offset = NUM_POSE + hand_idx * NUM_HAND
                        for j, lm in enumerate(hand_landmarks.landmark):
                            keypoints[offset + j] = [lm.x, lm.y, lm.z]
            
            # Fill face landmarks (75-552)
            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]
                offset = NUM_POSE + 2 * NUM_HAND  # 75
                for i, lm in enumerate(face_landmarks.landmark):
                    keypoints[offset + i] = [lm.x, lm.y, lm.z]
            
            all_keypoints.append(keypoints)
            
            # Clear frame from memory
            del frame, frame_rgb
        
        # Calculate detection statistics
        total_frames = len(all_keypoints)
        hand_detection_rate = (hand_detection_count / total_frames * 100) if total_frames > 0 else 0
        
        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(output_path),
            nodes=np.array(all_keypoints, dtype=np.float32),
            label=label,
            instance=instance_dir.name,
            hand_detection_rate=hand_detection_rate
        )
        
        # Clean up memory
        del all_keypoints
        
        with progress_lock:
            progress_stats['processed'] += 1
        
        return True, f"Hand detection: {hand_detection_rate:.1f}%"
        
    except Exception as e:
        with progress_lock:
            progress_stats['failed'] += 1
        return False, str(e)
        
    finally:
        # Always clean up MediaPipe resources
        hands_detector.close()
        pose_detector.close()
        face_detector.close()
        gc.collect()

def main():
    """Main processing function with conservative 4-worker threading"""
    
    print("🚀 Starting 553-node keypoint extraction (Multi-threaded version)")
    print("🔧 Using 4 workers for 4x speedup while maintaining stability")
    print("=" * 70)
    
    # Get all label directories
    label_dirs = [d for d in DATA_DIR.iterdir() if d.is_dir()]
    print(f"📁 Found {len(label_dirs)} label directories")
    
    # Collect all instances to process
    all_instances = []
    for label_dir in label_dirs:
        label = label_dir.name
        instance_dirs = [d for d in label_dir.iterdir() if d.is_dir()]
        for instance_dir in instance_dirs:
            all_instances.append((instance_dir, label))
    
    # Initialize progress tracking
    progress_stats['total'] = len(all_instances)
    progress_stats['start_time'] = time.time()
    
    print(f"🎯 Total instances to process: {len(all_instances)}")
    print(f"👥 Using 4 worker threads for parallel processing")
    print("=" * 70)
    
    # Process with ThreadPoolExecutor (conservative 4 workers)
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all tasks
        future_to_instance = {
            executor.submit(process_single_instance, instance_info): instance_info 
            for instance_info in all_instances
        }
        
        # Process completed tasks
        for future in as_completed(future_to_instance):
            instance_dir, label = future_to_instance[future]
            try:
                success, message = future.result()
                
                # Print progress every 50 completed instances
                if (progress_stats['processed'] + progress_stats['failed']) % 50 == 0:
                    print_progress()
                    
            except Exception as e:
                with progress_lock:
                    progress_stats['failed'] += 1
                print(f"❌ Unexpected error for {instance_dir.name}: {e}")
    
    # Final summary
    elapsed = time.time() - progress_stats['start_time']
    print(f"\n🏁 EXTRACTION COMPLETE!")
    print("=" * 70)
    print(f"✅ Successfully processed: {progress_stats['processed']}")
    print(f"❌ Failed: {progress_stats['failed']}")
    print(f"📊 Success rate: {(progress_stats['processed'] / progress_stats['total'] * 100):.1f}%")
    print(f"⏱️  Total time: {elapsed/3600:.1f} hours ({elapsed/60:.1f} minutes)")
    print(f"🚀 Average rate: {progress_stats['processed']/(elapsed/60):.1f} instances/minute")
    print(f"💾 Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
