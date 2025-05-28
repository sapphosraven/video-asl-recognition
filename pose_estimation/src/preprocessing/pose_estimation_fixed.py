"""
Memory-efficient 553-node keypoint extraction for ASL recognition.
Processes videos sequentially to avoid memory issues.
"""

import cv2
import mediapipe as mp
import os
import numpy as np
from pathlib import Path
import warnings
import gc

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

def process_single_instance(instance_dir, label):
    """Process a single video instance with memory efficiency"""
    
    images = sorted(instance_dir.glob('*.jpg'))
    if len(images) <= 1:
        print(f"⏭️  Skipping {instance_dir.name} - insufficient frames ({len(images)})")
        return False
    
    print(f"🎬 Processing {label}/{instance_dir.name} ({len(images)} frames)")
    
    # Check if already processed
    output_path = OUTPUT_DIR / label / f"{instance_dir.name}_keypoints.npz"
    if output_path.exists():
        print(f"   ✅ Already exists, skipping")
        return True
    
    # Initialize MediaPipe detectors with optimized settings
    hands_detector = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.2,  # Lower for ASL
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
    
    # Process frames one by one to save memory
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
            
            # Fill hand landmarks (33-74) - simplified approach
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
            
            # Progress indicator
            if (frame_idx + 1) % 20 == 0:
                print(f"   📊 Processed {frame_idx + 1}/{len(images)} frames")
        
        # Calculate detection statistics
        total_frames = len(all_keypoints)
        hand_detection_rate = (hand_detection_count / total_frames * 100) if total_frames > 0 else 0
        
        print(f"   🤲 Hand detection: {hand_detection_rate:.1f}% ({hand_detection_count}/{total_frames})")
        
        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(output_path),
            nodes=np.array(all_keypoints, dtype=np.float32),
            label=label,
            instance=instance_dir.name,
            hand_detection_rate=hand_detection_rate
        )
        
        print(f"   💾 Saved: {output_path}")
        
        # Clean up memory
        del all_keypoints
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error processing {instance_dir.name}: {e}")
        return False
        
    finally:
        # Always clean up MediaPipe resources
        hands_detector.close()
        pose_detector.close()
        face_detector.close()

def main():
    """Main processing function - sequential processing to avoid memory issues"""
    
    print("🚀 Starting 553-node keypoint extraction (Memory-efficient version)")
    print("=" * 70)
    
    # Get all label directories
    label_dirs = [d for d in DATA_DIR.iterdir() if d.is_dir()]
    print(f"📁 Found {len(label_dirs)} label directories")
    
    total_processed = 0
    total_failed = 0
    
    # Process each label sequentially
    for label_idx, label_dir in enumerate(label_dirs):
        label = label_dir.name
        print(f"\n📂 [{label_idx+1}/{len(label_dirs)}] Processing label: {label}")
        
        # Get all instances for this label
        instance_dirs = [d for d in label_dir.iterdir() if d.is_dir()]
        print(f"   🎯 Found {len(instance_dirs)} instances")
        
        # Process each instance
        for inst_idx, instance_dir in enumerate(instance_dirs):
            print(f"   [{inst_idx+1}/{len(instance_dirs)}] ", end="")
            
            success = process_single_instance(instance_dir, label)
            
            if success:
                total_processed += 1
            else:
                total_failed += 1
            
            # Force garbage collection every 10 instances
            if (inst_idx + 1) % 10 == 0:
                gc.collect()
                print(f"   🧹 Memory cleanup - Processed: {total_processed}, Failed: {total_failed}")
    
    # Final summary
    print(f"\n🏁 EXTRACTION COMPLETE!")
    print("=" * 70)
    print(f"✅ Successfully processed: {total_processed}")
    print(f"❌ Failed: {total_failed}")
    print(f"📊 Success rate: {(total_processed / (total_processed + total_failed) * 100):.1f}%")
    print(f"💾 Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
