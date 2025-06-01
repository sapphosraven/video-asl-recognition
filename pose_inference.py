import torch
import numpy as np
import json
import os
import logging
from typing import Union, Optional

logger = logging.getLogger(__name__)

# Import configuration utilities
try:
    from utils import config
    USE_CONFIG = True
except ImportError:
    USE_CONFIG = False
    logger.warning("Configuration utilities not available, using fallback settings")

# Global variables for caching
_pose_model = None
_idx_to_class = None
_openhands_available = None

def check_openhands_availability():
    """Check if OpenHands is available and cache the result"""
    global _openhands_available
    if _openhands_available is None:
        try:
            import openhands.models
            _openhands_available = True
            logger.info("OpenHands library is available")
        except ImportError:
            _openhands_available = False
            logger.warning("OpenHands library not available. Install with: pip install git+https://github.com/AI4Bharat/OpenHands.git")
    return _openhands_available

def load_idx_to_class_mapping():
    """Load the class index to word mapping from the wordlevelrecogntion directory"""
    global _idx_to_class
    if _idx_to_class is not None:
        return _idx_to_class
    
    # Try WLASL300 class mapping first
    idx_file_300 = os.path.join(os.path.dirname(__file__), "wordlevelrecogntion", "class_map_wlasl300.json")
    idx_file = os.path.join(os.path.dirname(__file__), "wordlevelrecogntion", "class_map.json")
    
    # Prefer WLASL300 mapping if available
    if os.path.exists(idx_file_300):
        logger.info("Using WLASL300 class mapping")
        with open(idx_file_300, 'r', encoding='utf-8') as f:
            _idx_to_class = json.load(f)
    elif os.path.exists(idx_file):
        logger.info("Using original class mapping")
        with open(idx_file, 'r', encoding='utf-8') as f:
            _idx_to_class = json.load(f)
    else:
        logger.error(f"Class mapping file not found at {idx_file_300} or {idx_file}")
        raise FileNotFoundError(f"Class mapping file not found at {idx_file_300} or {idx_file}")
    
    logger.info(f"Loaded class mapping with {len(_idx_to_class)} classes")
    return _idx_to_class

def get_num_classes():
    """Get the number of classes from the class mapping"""
    idx_to_class = load_idx_to_class_mapping()
    return len(idx_to_class)

def load_pose_model(checkpoint_path: str, device: Union[str, torch.device] = "cpu"):
    """
    Load the OpenHands T-GCN model for pose-based word recognition.
    
    Args:
        checkpoint_path: Path to the pretrained model checkpoint
        device: Device to load the model on
        
    Returns:
        The loaded T-GCN model or None if OpenHands is not available
    """
    global _pose_model
    
    # Check if OpenHands is available
    if not check_openhands_availability():
        logger.error("Cannot load pose model: OpenHands library not available")
        return None
    
    try:
        # Import OpenHands components
        from openhands.models import PoseNet
        
        # Get number of classes dynamically
        num_classes = get_num_classes()
        logger.info(f"Initializing T-GCN model with {num_classes} classes")
        
        # Get model configuration
        if USE_CONFIG:
            model_config = config.get_pose_model_config()
            model_name = model_config.get('name', 'tgcn')
            num_joints = model_config.get('num_joints', 25)
            in_channels = model_config.get('in_channels', 3)
        else:
            model_name = 'tgcn'
            num_joints = 25
            in_channels = 3
        
        # Initialize the model
        model = PoseNet(
            model_name=model_name, 
            num_joints=num_joints, 
            in_channels=in_channels, 
            num_classes=num_classes
        )
        
        # Load checkpoint
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint file not found at {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Try to load state dict, using strict=False for flexibility
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            logger.warning(f"Strict loading failed, trying with strict=False: {e}")
            model.load_state_dict(state_dict, strict=False)
        
        model.to(device)
        model.eval()
        
        _pose_model = model
        logger.info("T-GCN model loaded successfully")
        return model
        
    except ImportError as e:
        logger.error(f"Failed to import OpenHands: {e}")
        logger.error("Please install OpenHands: pip install git+https://github.com/AI4Bharat/OpenHands.git@main#egg=OpenHands")
        return None
    except Exception as e:
        logger.error(f"Failed to load pose model: {e}")
        return None

def reshape_553_to_25(arr_553):
    """
    Reshape MediaPipe keypoints from 553 dimensions to 25 body joints.
    
    Args:
        arr_553: Array of shape (..., 553) or (..., 553, 3)
        
    Returns:
        Array of shape (..., 25, 3) containing only body pose landmarks
    """
    # If the array is flattened (553,), reshape to (553, 3)
    if arr_553.ndim == 1:
        arr_553 = arr_553.reshape(553, 3)
    elif arr_553.shape[-1] == 553:
        # If shape is (T, 553), reshape to (T, 553, 3)
        arr_553 = arr_553.reshape(*arr_553.shape[:-1], 553, 3)
    
    # MediaPipe landmarks order: pose (33), left_hand (21), right_hand (21), face (468)
    # We want only the first 25 pose landmarks (excluding some pose landmarks)
    # Take the first 25 landmarks which should be the main body pose points
    return arr_553[..., :25, :]

def pad_or_truncate_sequence(keypoint_array: np.ndarray, target_length: int = 60):
    """
    Pad or truncate a keypoint sequence to the target length.
    
    Args:
        keypoint_array: Array of shape (T, 25, 3)
        target_length: Target sequence length (default: 60)
        
    Returns:
        Array of shape (target_length, 25, 3)
    """
    T, V, C = keypoint_array.shape
    
    if T >= target_length:
        # Truncate by sampling uniformly
        indices = np.linspace(0, T-1, target_length, dtype=int)
        return keypoint_array[indices]
    else:
        # Pad by repeating the last frame
        padding = np.repeat(keypoint_array[-1:], target_length - T, axis=0)
        return np.concatenate([keypoint_array, padding], axis=0)

def predict_word_from_pose(model, keypoint_array: np.ndarray, device: Union[str, torch.device] = "cpu", 
                          min_confidence: float = None):
    """
    Predict a word from pose keypoints using the T-GCN model.
    
    Args:
        model: The loaded T-GCN model (or None if not available)
        keypoint_array: Numpy array of shape (T, 25, 3) containing pose keypoints
        device: Device to run inference on
        min_confidence: Minimum confidence threshold for prediction
        
    Returns:
        Predicted word string, or "unknown" if confidence is too low or model unavailable
    """
    # Check if model is available
    if model is None:
        logger.warning("Pose model not available, returning 'unknown'")
        return "unknown"
    
    # Get confidence threshold from config or use provided value
    if min_confidence is None:
        if USE_CONFIG:
            min_confidence = config.get('model_config.pose_model.min_confidence', 0.25)
        else:
            min_confidence = 0.25
    
    try:
        # Validate input shape
        if keypoint_array.ndim != 3 or keypoint_array.shape[1] != 25 or keypoint_array.shape[2] != 3:
            raise ValueError(f"Expected keypoint array shape (T, 25, 3), got {keypoint_array.shape}")
        
        # Get target length from config or use default
        if USE_CONFIG:
            target_len = config.get('model_config.pose_model.target_sequence_length', 60)
        else:
            target_len = 60
        
        # Pad/truncate to target length
        processed_keypoints = pad_or_truncate_sequence(keypoint_array, target_len)
        
        # Convert to torch tensor and add batch dimension
        x = torch.tensor(processed_keypoints, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 60, 25, 3)
        
        # Run inference
        with torch.no_grad():
            logits = model(x)  # (1, num_classes)
            probs = torch.nn.functional.softmax(logits, dim=1)[0]  # (num_classes,)
            
            # Check if model is making meaningful predictions
            prob_std = probs.std().item()
            if prob_std < 0.01:
                logger.warning(f"Model not discriminating well (std={prob_std:.5f})")
                return "unknown"
            
            # Get prediction and confidence
            confidence, pred_idx = torch.max(probs, dim=0)
            pred_index = pred_idx.item()
            
            # Apply confidence threshold
            if confidence.item() < min_confidence:
                logger.warning(f"Low confidence prediction ({confidence.item()*100:.2f}%)")
                return "unknown"
            
            # Map index to class name
            idx_to_class = load_idx_to_class_mapping()
            predicted_word = idx_to_class.get(str(pred_index), "unknown")
            
            logger.info(f"Pose prediction: '{predicted_word}' (confidence: {confidence.item()*100:.2f}%)")
            return predicted_word
            
    except Exception as e:
        logger.error(f"Error in pose-based prediction: {str(e)}")
        return "unknown"

def extract_keypoints_from_video_clip(clip_path: str):
    """
    Extract pose keypoints from a video clip using MediaPipe.
    
    Args:
        clip_path: Path to the video clip
        
    Returns:
        Numpy array of shape (T, 25, 3) containing pose keypoints
    """
    try:
        import cv2
        import mediapipe as mp
        
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        cap = cv2.VideoCapture(clip_path)
        keypoints_list = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Extract pose landmarks (33 points)
                landmarks = results.pose_landmarks.landmark
                pose_keypoints = []
                
                for landmark in landmarks[:25]:  # Take first 25 landmarks
                    pose_keypoints.append([landmark.x, landmark.y, landmark.z])
                
                keypoints_list.append(pose_keypoints)
            else:
                # If no pose detected, use zero keypoints
                keypoints_list.append([[0.0, 0.0, 0.0]] * 25)
        
        cap.release()
        pose.close()
        
        if not keypoints_list:
            raise ValueError(f"No keypoints extracted from {clip_path}")
        
        return np.array(keypoints_list, dtype=np.float32)  # Shape: (T, 25, 3)
        
    except Exception as e:
        logger.error(f"Error extracting keypoints from {clip_path}: {e}")
        raise

def get_pose_model():
    """Get the cached pose model instance"""
    global _pose_model
    return _pose_model

def is_pose_model_loaded():
    """Check if the pose model is loaded"""
    return _pose_model is not None
