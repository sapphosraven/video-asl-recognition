"""
Simple hand detection test that works around MediaPipe permission issues.
Tests hand detection on a few sample images to verify ASL hand detection.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def test_hand_detection_simple():
    """Simple test of hand detection on sample ASL images"""
    
    print("🧪 SIMPLE HAND DETECTION TEST FOR ASL")
    print("=" * 50)
    
    # Find sample images
    data_dir = Path(r'f:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\data\processed')
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return False
    
    # Get sample images from first few labels
    sample_images = []
    for label_dir in list(data_dir.iterdir())[:3]:  # Test 3 labels
        if label_dir.is_dir():
            for instance_dir in list(label_dir.iterdir())[:2]:  # 2 instances per label
                if instance_dir.is_dir():
                    images = list(instance_dir.glob('*.jpg'))
                    if images:
                        # Take middle frame (often best for hand detection)
                        mid_frame = images[len(images)//2]
                        sample_images.append((mid_frame, label_dir.name, instance_dir.name))
    
    if not sample_images:
        print("❌ No sample images found!")
        return False
    
    print(f"📁 Testing {len(sample_images)} sample images...")
    
    # Initialize MediaPipe with basic settings (avoid model complexity issues)
    try:
        mp_hands = mp.solutions.hands
        hands_detector = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.2,  # Lower threshold
            min_tracking_confidence=0.2
        )
        
        mp_pose = mp.solutions.pose
        pose_detector = mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.3
        )
        
        print("✅ MediaPipe initialized successfully")
        
    except Exception as e:
        print(f"❌ MediaPipe initialization failed: {e}")
        return False
    
    # Test detection on samples
    results = []
    
    for i, (img_path, label, instance) in enumerate(sample_images):
        print(f"\n{i+1}. Testing: {label}/{instance}")
        
        try:
            # Load image
            frame = cv2.imread(str(img_path))
            if frame is None:
                print("   ❌ Failed to load image")
                continue
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            print(f"   📐 Image size: {w}x{h}")
            
            # Test hand detection
            hand_results = hands_detector.process(frame_rgb)
            pose_results = pose_detector.process(frame_rgb)
            
            # Analyze results
            hands_detected = hand_results.multi_hand_landmarks is not None
            num_hands = len(hand_results.multi_hand_landmarks) if hands_detected else 0
            pose_detected = pose_results.pose_landmarks is not None
            
            print(f"   🤲 Hands: {'✅' if hands_detected else '❌'} ({num_hands} hands)")
            print(f"   🧍 Pose: {'✅' if pose_detected else '❌'}")
            
            # Get hand details
            if hands_detected:
                for j, (hand_landmarks, handedness) in enumerate(zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness)):
                    hand_label = handedness.classification[0].label
                    confidence = handedness.classification[0].score
                    print(f"      Hand {j+1}: {hand_label} (confidence: {confidence:.3f})")
            
            results.append({
                'label': label,
                'instance': instance,
                'hands_detected': hands_detected,
                'num_hands': num_hands,
                'pose_detected': pose_detected
            })
            
        except Exception as e:
            print(f"   ❌ Error processing image: {e}")
            continue
    
    # Cleanup
    hands_detector.close()
    pose_detector.close()
    
    # Summary
    print(f"\n📊 SUMMARY RESULTS:")
    print("=" * 50)
    
    if results:
        total_tests = len(results)
        hand_success = sum(1 for r in results if r['hands_detected'])
        pose_success = sum(1 for r in results if r['pose_detected'])
        total_hands = sum(r['num_hands'] for r in results)
        
        hand_rate = (hand_success / total_tests) * 100
        pose_rate = (pose_success / total_tests) * 100
        
        print(f"Total tests: {total_tests}")
        print(f"Hand detection rate: {hand_rate:.1f}% ({hand_success}/{total_tests})")
        print(f"Pose detection rate: {pose_rate:.1f}% ({pose_success}/{total_tests})")
        print(f"Total hands detected: {total_hands}")
        print(f"Average hands per image: {total_hands/total_tests:.1f}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if hand_rate >= 50:
            print("🎉 EXCELLENT: Hand detection rate is good for ASL!")
            print("   ✅ Proceed with keypoint extraction")
        elif hand_rate >= 30:
            print("✅ GOOD: Hand detection rate is acceptable")
            print("   💡 Interpolation will help fill missing frames")
            print("   ✅ Proceed with keypoint extraction")
        elif hand_rate >= 10:
            print("⚠️  MODERATE: Hand detection needs improvement")
            print("   💡 Lower confidence threshold further (0.1)")
            print("   💡 Ensure good lighting in videos")
            print("   ⚠️  Proceed but expect lower accuracy")
        else:
            print("❌ POOR: Hand detection rate is too low!")
            print("   🔧 Try these fixes:")
            print("      - Lower confidence to 0.1")
            print("      - Check video quality")
            print("      - Verify hand visibility in frames")
            print("   ❌ Fix before proceeding")
        
        return hand_rate >= 30  # Return True if acceptable
    
    else:
        print("❌ No successful tests!")
        return False

if __name__ == "__main__":
    success = test_hand_detection_simple()
    
    if success:
        print(f"\n🚀 READY TO PROCEED!")
        print("   Run: python pose_estimation_mediapipe.py")
    else:
        print(f"\n⚠️  NEED TO FIX HAND DETECTION FIRST!")
        print("   Check MediaPipe settings and video quality")
