"""
TGCN inference module for ASL word recognition.
Uses pretrained TGCN model from OpenHands library.
"""

import torch
import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path
import json
from typing import List, Tuple, Dict, Any

class TGCNInference:
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Load class mapping for WLASL300
        self.load_class_mapping()
        
    def load_model(self):
        """Load pretrained TGCN model from OpenHands."""
        try:
            from openhands.models.builder import load_model
            
            print(f"Loading TGCN model on device: {self.device}")
            
            # Load pretrained TGCN for word-level recognition
            self.model = load_model(
                model_name="TGCN",                      # Model architecture
                modality="pose",                        # Modality (pose, rgb, etc.)
                dataset="phoenix14",                    # Target dataset used to train
                checkpoint="word",                      # Type of checkpoint
                pretrained=True,                        # Load pretrained weights
            )
            
            self.model.eval()
            self.model.to(self.device)
            
            print("✅ TGCN model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load TGCN model: {e}")
            return False
    
    def load_class_mapping(self):
        """Load class index to word mapping."""
        # For now, use the CNN class mapping as a reference
        # TODO: Update with actual TGCN class mapping when available
        class_map_path = Path("wordlevelrecogntion/class_map_correct.json")
        if class_map_path.exists():
            with open(class_map_path, 'r') as f:
                class_map = json.load(f)
            
            self.idx_to_word = {}
            for idx_str, word in class_map.items():
                self.idx_to_word[int(idx_str)] = word
        else:
            print("⚠️ Class mapping not found, using dummy mapping")
            self.idx_to_word = {i: f"word_{i}" for i in range(300)}
    
    def extract_pose_keypoints(self, video_path: str) -> np.ndarray:
        """
        Extract pose keypoints from video using MediaPipe.
        
        Returns:
            np.ndarray: Shape [T, J, C] where T=frames, J=joints, C=coordinates
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        keypoints_sequence = []
        
        with self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False
        ) as holistic:
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame
                results = holistic.process(frame_rgb)
                
                # Extract keypoints
                frame_keypoints = self.extract_frame_keypoints(results)
                keypoints_sequence.append(frame_keypoints)
        
        cap.release()
        
        if len(keypoints_sequence) == 0:
            raise ValueError("No keypoints extracted from video")
        
        # Convert to numpy array: [T, J, C]
        keypoints_array = np.array(keypoints_sequence)
        print(f"Extracted keypoints shape: {keypoints_array.shape}")
        
        return keypoints_array
    
    def extract_frame_keypoints(self, results) -> np.ndarray:
        """
        Extract keypoints from a single frame's MediaPipe results.
        
        We'll focus on hand keypoints for ASL recognition.
        Each hand has 21 keypoints with (x, y, z) coordinates.
        Total: 42 keypoints * 3 coordinates = 126 features
        
        But we'll reshape to [42, 3] to match TGCN expected format.
        """
        # Initialize keypoints array for both hands
        # 21 keypoints per hand, 2 hands = 42 total keypoints
        # Each keypoint has (x, y, z) coordinates
        keypoints = np.zeros((42, 3))
        
        # Left hand keypoints (0-20)
        if results.left_hand_landmarks:
            for i, landmark in enumerate(results.left_hand_landmarks.landmark):
                keypoints[i] = [landmark.x, landmark.y, landmark.z]
        
        # Right hand keypoints (21-41)
        if results.right_hand_landmarks:
            for i, landmark in enumerate(results.right_hand_landmarks.landmark):
                keypoints[21 + i] = [landmark.x, landmark.y, landmark.z]
        
        return keypoints
    
    def preprocess_for_tgcn(self, keypoints: np.ndarray, target_frames: int = 32) -> torch.Tensor:
        """
        Preprocess keypoints for TGCN input.
        
        Args:
            keypoints: [T, J, C] array
            target_frames: Target number of frames
            
        Returns:
            torch.Tensor: [1, target_frames, J, C] ready for TGCN
        """
        T, J, C = keypoints.shape
        
        # Resample to target number of frames
        if T != target_frames:
            indices = np.linspace(0, T - 1, target_frames, dtype=int)
            keypoints = keypoints[indices]
        
        # Convert to tensor and add batch dimension
        keypoints_tensor = torch.from_numpy(keypoints).float()
        keypoints_tensor = keypoints_tensor.unsqueeze(0)  # Add batch dimension
        
        # Move to device
        keypoints_tensor = keypoints_tensor.to(self.device)
        
        print(f"TGCN input shape: {keypoints_tensor.shape}")
        return keypoints_tensor
    
    def predict_word(self, video_path: str) -> Dict[str, Any]:
        """
        Predict word from video using TGCN.
        
        Returns:
            dict: Prediction results with word, confidence, etc.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Extract pose keypoints
            keypoints = self.extract_pose_keypoints(video_path)
            
            # Preprocess for TGCN
            input_tensor = self.preprocess_for_tgcn(keypoints)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # Get top predictions
                top_probs, top_indices = torch.topk(probabilities, 5)
                
                predictions = []
                for i in range(5):
                    class_idx = top_indices[0][i].item()
                    confidence = top_probs[0][i].item()
                    word = self.idx_to_word.get(class_idx, f"unknown_{class_idx}")
                    
                    predictions.append({
                        'word': word,
                        'confidence': confidence,
                        'class_idx': class_idx
                    })
                
                return {
                    'success': True,
                    'predictions': predictions,
                    'input_shape': input_tensor.shape,
                    'output_shape': outputs.shape
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'predictions': []
            }

def test_tgcn_loading():
    """Test TGCN model loading and basic functionality."""
    print("=== Testing TGCN Model Loading ===")
    
    tgcn = TGCNInference()
    
    # Test model loading
    if not tgcn.load_model():
        print("❌ Failed to load model")
        return False
    
    # Test with dummy data
    print("\nTesting with dummy pose data...")
    try:
        # Create dummy pose keypoints: [B, T, J, C]
        # B=1, T=32 frames, J=42 joints (hand keypoints), C=3 coordinates
        dummy_input = torch.randn(1, 32, 42, 3).to(tgcn.device)
        
        with torch.no_grad():
            outputs = tgcn.model(dummy_input)
            print(f"✅ Model forward pass successful!")
            print(f"Input shape: {dummy_input.shape}")
            print(f"Output shape: {outputs.shape}")
            
            # Get predictions
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, 3)
            
            print(f"Top 3 predictions:")
            for i in range(3):
                class_idx = top_indices[0][i].item()
                confidence = top_probs[0][i].item()
                word = tgcn.idx_to_word.get(class_idx, f"unknown_{class_idx}")
                print(f"  {i+1}. {word} (class {class_idx}): {confidence:.4f}")
                
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_video_processing():
    """Test video processing with actual videos."""
    print("\n=== Testing Video Processing ===")
    
    tgcn = TGCNInference()
    
    if not tgcn.load_model():
        return False
    
    # Test with a sample video
    video_path = "uploads/00414.mp4"  # Should be an "about" video
    
    if not Path(video_path).exists():
        print(f"❌ Test video not found: {video_path}")
        return False
    
    print(f"Processing video: {video_path}")
    result = tgcn.predict_word(video_path)
    
    if result['success']:
        print("✅ Video processing successful!")
        print(f"Input shape: {result['input_shape']}")
        print(f"Output shape: {result['output_shape']}")
        print("Top 5 predictions:")
        for i, pred in enumerate(result['predictions']):
            print(f"  {i+1}. {pred['word']}: {pred['confidence']:.4f}")
    else:
        print(f"❌ Video processing failed: {result['error']}")
        return False
    
    return True

if __name__ == "__main__":
    print("TGCN Inference Testing")
    print("=" * 50)
    
    # Test 1: Model loading
    if test_tgcn_loading():
        print("\n" + "=" * 50)
        # Test 2: Video processing
        test_video_processing()
    
    print("\nTesting complete!")
