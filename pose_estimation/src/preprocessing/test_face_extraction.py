"""
Test script to verify face landmark extraction is working
"""
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

def test_face_landmarks():
    """Test face landmark extraction on a single image"""
    
    # Find a test image (take the first image from any class)
    data_dir = Path(r'f:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\data\processed')
    
    # Find first available image
    test_image = None
    for class_dir in data_dir.iterdir():
        if class_dir.is_dir():
            for instance_dir in class_dir.iterdir():
                if instance_dir.is_dir():
                    images = list(instance_dir.glob('*.jpg'))
                    if images:
                        test_image = images[0]
                        break
            if test_image:
                break
    
    if not test_image:
        print("No test image found!")
        return False
    
    print(f"Testing with image: {test_image}")
    
    # Load image
    frame = cv2.imread(str(test_image))
    if frame is None:
        print("Failed to load image!")
        return False
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Initialize detectors
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh, mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    ) as hands, mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.5
    ) as pose:
        
        # Process image
        face_results = face_mesh.process(frame_rgb)
        hand_results = hands.process(frame_rgb)
        pose_results = pose.process(frame_rgb)
        
        # Check results
        print(f"Image dimensions: {frame.shape}")
        print(f"Face detected: {face_results.multi_face_landmarks is not None}")
        print(f"Hands detected: {hand_results.multi_hand_landmarks is not None}")
        print(f"Pose detected: {pose_results.pose_landmarks is not None}")
        
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            print(f"Number of face landmarks: {len(face_landmarks.landmark)}")
            
        if hand_results.multi_hand_landmarks:
            print(f"Number of hands detected: {len(hand_results.multi_hand_landmarks)}")
            
        if pose_results.pose_landmarks:
            print(f"Number of pose landmarks: {len(pose_results.pose_landmarks.landmark)}")
        
        # Calculate total nodes
        total_nodes = 33 + 42 + 468  # pose + hands + face
        print(f"Total expected nodes: {total_nodes}")
        
        return True

if __name__ == "__main__":
    test_face_landmarks()
