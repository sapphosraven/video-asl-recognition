"""
Improved normalization and preprocessing for ASL TGCN based on successful implementations.

This module implements state-of-the-art preprocessing techniques found in literature:
1. Spatial anchoring (relative to body center)
2. Temporal smoothing and interpolation 
3. Scale normalization
4. Data augmentation
5. Improved graph connectivity

References:
- "Pose-based Sign Language Recognition using GCN and BERT" - 87.60% on WLASL-100
- "Preprocessing Mediapipe Keypoints with Keypoint Reconstruction and Anchors"
- Original WLASL Pose-TGCN implementation
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

class ImprovedPoseNormalizer:
    """Advanced pose keypoint normalizer based on successful TGCN implementations"""
    
    def __init__(self):
        # MediaPipe pose landmarks indices for anchoring
        self.NOSE_IDX = 0
        self.LEFT_SHOULDER_IDX = 11  
        self.RIGHT_SHOULDER_IDX = 12
        self.LEFT_HIP_IDX = 23
        self.RIGHT_HIP_IDX = 24
        
        # MediaPipe keypoint constants for 553-node architecture
        self.NUM_POSE = 33
        self.NUM_HAND = 21  
        self.NUM_FACE = 478  # MediaPipe face mesh provides 478 landmarks
        self.TOTAL_NODES = self.NUM_POSE + 2 * self.NUM_HAND + self.NUM_FACE  # 553
        
        # Keypoint ranges
        self.POSE_RANGE = range(0, 33)
        self.LEFT_HAND_RANGE = range(33, 54)  
        self.RIGHT_HAND_RANGE = range(54, 75)
        self.FACE_RANGE = range(75, 553)  # Face landmarks start at index 75
        
    def calculate_body_center(self, pose_keypoints):
        """Calculate body center from shoulders and hips for anchoring"""
        try:
            # Get key body points
            left_shoulder = pose_keypoints[:, self.LEFT_SHOULDER_IDX, :2]
            right_shoulder = pose_keypoints[:, self.RIGHT_SHOULDER_IDX, :2]  
            left_hip = pose_keypoints[:, self.LEFT_HIP_IDX, :2]
            right_hip = pose_keypoints[:, self.RIGHT_HIP_IDX, :2]
            
            # Calculate center as mean of shoulders and hips
            shoulder_center = (left_shoulder + right_shoulder) / 2
            hip_center = (left_hip + right_hip) / 2
            body_center = (shoulder_center + hip_center) / 2
            
            return body_center  # Shape: [seq_len, 2]
        except:
            # Fallback to nose if body points are missing
            return pose_keypoints[:, self.NOSE_IDX, :2]
    
    def calculate_body_scale(self, pose_keypoints):
        """Calculate body scale for size normalization"""
        try:
            # Calculate shoulder width
            left_shoulder = pose_keypoints[:, self.LEFT_SHOULDER_IDX, :2]
            right_shoulder = pose_keypoints[:, self.RIGHT_SHOULDER_IDX, :2]
            shoulder_width = np.linalg.norm(left_shoulder - right_shoulder, axis=1)
            
            # Calculate torso height  
            shoulder_center = (left_shoulder + right_shoulder) / 2
            left_hip = pose_keypoints[:, self.LEFT_HIP_IDX, :2]
            right_hip = pose_keypoints[:, self.RIGHT_HIP_IDX, :2]
            hip_center = (left_hip + right_hip) / 2
            torso_height = np.linalg.norm(shoulder_center - hip_center, axis=1)
            
            # Use shoulder width as primary scale reference
            scale = shoulder_width + 1e-6  # Avoid division by zero
            scale = np.median(scale)  # Use median for robustness
            
            return max(scale, 0.1)  # Minimum scale threshold
        except:
            return 1.0  # Default scale
    
    def smooth_temporal_sequence(self, keypoints, sigma=1.0):
        """Apply Gaussian smoothing along temporal dimension"""
        smoothed = np.zeros_like(keypoints)
        
        for node in range(keypoints.shape[1]):
            for coord in range(keypoints.shape[2]):
                sequence = keypoints[:, node, coord]
                # Only smooth non-zero sequences
                if np.sum(np.abs(sequence)) > 0:
                    smoothed[:, node, coord] = gaussian_filter1d(sequence, sigma=sigma)
                else:
                    smoothed[:, node, coord] = sequence
                    
        return smoothed
    
    def interpolate_missing_keypoints(self, keypoints):
        """Interpolate missing keypoints (zeros) using neighboring frames"""
        seq_len, num_nodes, num_coords = keypoints.shape
        interpolated = keypoints.copy()
        
        for node in range(num_nodes):
            for coord in range(num_coords):
                sequence = interpolated[:, node, coord]
                
                # Find valid (non-zero) indices
                valid_indices = np.where(np.abs(sequence) > 1e-6)[0]
                
                if len(valid_indices) > 1:
                    # Interpolate missing values
                    f = interp1d(valid_indices, sequence[valid_indices], 
                               kind='linear', fill_value='extrapolate')
                    
                    # Fill in missing values
                    missing_indices = np.where(np.abs(sequence) <= 1e-6)[0]
                    if len(missing_indices) > 0:
                        interpolated[missing_indices, node, coord] = f(missing_indices)
        
        return interpolated
    
    def normalize_pose_sequence(self, keypoints):
        """
        Main normalization function implementing best practices from literature
        
        Args:
            keypoints: [seq_len, num_nodes, 3] - raw MediaPipe keypoints
            
        Returns:
            normalized_keypoints: [seq_len, num_nodes, 3] - normalized keypoints
        """
        if keypoints.shape[0] == 0:
            return keypoints
            
        # Step 1: Interpolate missing keypoints
        keypoints = self.interpolate_missing_keypoints(keypoints)
        
        # Step 2: Temporal smoothing
        keypoints = self.smooth_temporal_sequence(keypoints, sigma=0.8)
        
        # Step 3: Calculate body center and scale for each frame
        body_centers = self.calculate_body_center(keypoints)  # [seq_len, 2]
        body_scale = self.calculate_body_scale(keypoints)     # scalar
        
        normalized = keypoints.copy()
        
        # Step 4: Spatial normalization (center and scale)
        for t in range(keypoints.shape[0]):
            # Center all keypoints relative to body center
            normalized[t, :, 0] = (keypoints[t, :, 0] - body_centers[t, 0]) / body_scale
            normalized[t, :, 1] = (keypoints[t, :, 1] - body_centers[t, 1]) / body_scale
            
            # Z-coordinate: normalize to reasonable range
            normalized[t, :, 2] = keypoints[t, :, 2] / body_scale
        
        # Step 5: Clip to reasonable range
        normalized = np.clip(normalized, -3.0, 3.0)
        
        # Step 6: Handle any remaining NaN/inf values
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        
        return normalized

def create_improved_pose_dataset_class():
    """Create an improved dataset class with advanced normalization"""
    
    class ImprovedPoseSequenceDataset:
        def __init__(self, data_dir, max_seq_len=50, split='train', test_size=0.2, random_state=42, 
                     use_subset=True, max_classes=100):
            
            import torch
            from torch.utils.data import Dataset
            import numpy as np
            import glob
            import os
            from sklearn.model_selection import train_test_split
            from collections import defaultdict
            import json
            
            self.data_dir = data_dir
            self.max_seq_len = max_seq_len
            self.split = split
            self.normalizer = ImprovedPoseNormalizer()
            
            print(f"Loading IMPROVED dataset from: {data_dir}")
            
            # Find all NPZ files
            self.files = []
            self.labels = []
            self.word_to_idx = {}
            self.idx_to_word = {}
            
            # Scan directory structure
            word_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
            word_dirs = sorted(word_dirs)
            
            # Use subset for better training (successful papers often use WLASL-100)
            if use_subset and len(word_dirs) > max_classes:
                print(f"🎯 Using subset of {max_classes} classes for better training")
                # Select most common words (better data quality)
                word_counts = []
                for word in word_dirs:
                    word_dir = os.path.join(data_dir, word)
                    count = len(glob.glob(os.path.join(word_dir, "*.npz")))
                    word_counts.append((word, count))
                
                # Sort by count and take top classes
                word_counts.sort(key=lambda x: x[1], reverse=True)
                word_dirs = [word for word, count in word_counts[:max_classes]]
                print(f"Selected top {len(word_dirs)} classes with most samples")
            
            print(f"Found {len(word_dirs)} word categories")
            
            # Build label mapping
            for idx, word in enumerate(word_dirs):
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
            
            self.num_classes = len(self.word_to_idx)
            print(f"Number of classes: {self.num_classes}")
            
            # Collect all files with labels
            file_label_pairs = []
            for word, label_idx in self.word_to_idx.items():
                word_dir = os.path.join(data_dir, word)
                npz_files = glob.glob(os.path.join(word_dir, "*.npz"))
                
                # Quality filter: only use files with sufficient frames
                for file_path in npz_files:
                    try:
                        data = np.load(file_path)
                        if 'nodes' in data:
                            keypoints = data['nodes']
                            if len(keypoints.shape) == 3 and keypoints.shape[0] >= 10:  # At least 10 frames
                                file_label_pairs.append((file_path, label_idx))
                    except:
                        continue
            
            print(f"Total valid files found: {len(file_label_pairs)}")
            
            if len(file_label_pairs) == 0:
                raise ValueError(f"No valid NPZ files found in {data_dir}")
            
            # Split into train/test
            files, labels = zip(*file_label_pairs)
            
            if len(files) > 1:
                train_files, test_files, train_labels, test_labels = train_test_split(
                    files, labels, test_size=test_size, random_state=random_state, 
                    stratify=labels
                )
            else:
                train_files, test_files = files, files
                train_labels, test_labels = labels, labels
            
            if split == 'train':
                self.files = list(train_files)
                self.labels = list(train_labels)
            else:
                self.files = list(test_files)
                self.labels = list(test_labels)
            
            print(f"{split.upper()} split: {len(self.files)} files")
            
            # Class distribution
            class_counts = defaultdict(int)
            for label in self.labels:
                class_counts[self.idx_to_word[label]] += 1
            
            print(f"Class distribution in {split} (showing top 10):")
            for i, (word, count) in enumerate(sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]):
                print(f"  {word}: {count} samples")
        
        def __len__(self):
            return len(self.files)
        
        def __getitem__(self, idx):
            file_path = self.files[idx]
            label = self.labels[idx]
            
            try:
                # Load NPZ file
                data = np.load(file_path)
                
                # Handle different possible keys
                if 'nodes' in data:
                    keypoints = data['nodes']  # Shape: (seq_len, num_nodes, 3)
                elif 'keypoints' in data:
                    keypoints = data['keypoints']
                else:
                    raise ValueError(f"No keypoint data found in {file_path}")
                
                # Ensure proper shape
                if len(keypoints.shape) != 3:
                    raise ValueError(f"Invalid keypoint shape: {keypoints.shape}")
                
                seq_len, num_nodes, num_features = keypoints.shape
                
                # Apply IMPROVED normalization
                keypoints = self.normalizer.normalize_pose_sequence(keypoints)
                
                # Handle sequence length with better resampling
                if seq_len > self.max_seq_len:
                    # Uniform resampling (better than just truncating)
                    indices = np.linspace(0, seq_len - 1, self.max_seq_len, dtype=int)
                    keypoints = keypoints[indices]
                elif seq_len < self.max_seq_len:
                    # Pad sequence
                    padding = np.zeros((self.max_seq_len - seq_len, num_nodes, num_features))
                    keypoints = np.concatenate([keypoints, padding], axis=0)
                
                # Convert to tensor
                keypoints = keypoints.astype(np.float32)
                
                return torch.tensor(keypoints, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                # Return zero tensor as fallback (updated for 553 nodes)
                zero_keypoints = torch.zeros((self.max_seq_len, 553, 3), dtype=torch.float32)
                return zero_keypoints, torch.tensor(label, dtype=torch.long)
    
    return ImprovedPoseSequenceDataset

def create_improved_graph_connectivity():
    """Create improved graph connectivity for 553 nodes (pose + hands + face)"""
    
    # MediaPipe keypoint indices for 553-node architecture
    NUM_POSE, NUM_HAND, NUM_FACE = 33, 21, 478
    TOTAL_NODES = NUM_POSE + 2 * NUM_HAND + NUM_FACE  # 553
    
    # Import MediaPipe connections
    from mediapipe.python.solutions import pose_connections, hands_connections
    POSE_CONNECTIONS = pose_connections.POSE_CONNECTIONS
    HAND_CONNECTIONS = hands_connections.HAND_CONNECTIONS
    
    edges = set()
    
    # 1. Original MediaPipe connections
    # Pose connections (0-32)
    for u, v in POSE_CONNECTIONS:
        edges.add((u, v))
        edges.add((v, u))
    
    # Left hand connections (33-53)
    off1 = NUM_POSE
    for u, v in HAND_CONNECTIONS:
        edges.add((off1 + u, off1 + v))
        edges.add((off1 + v, off1 + u))
    
    # Right hand connections (54-74)
    off2 = NUM_POSE + NUM_HAND
    for u, v in HAND_CONNECTIONS:
        edges.add((off2 + u, off2 + v))
        edges.add((off2 + v, off2 + u))
    
    # 2. IMPROVED: Add hand-wrist connections
    LEFT_WRIST = 15   # MediaPipe pose wrist
    RIGHT_WRIST = 16
    LEFT_HAND_BASE = 33   # First left hand keypoint
    RIGHT_HAND_BASE = 54  # First right hand keypoint
    
    # Connect wrists to hand bases
    edges.add((LEFT_WRIST, LEFT_HAND_BASE))
    edges.add((LEFT_HAND_BASE, LEFT_WRIST))
    edges.add((RIGHT_WRIST, RIGHT_HAND_BASE))
    edges.add((RIGHT_HAND_BASE, RIGHT_WRIST))
    
    # 3. IMPROVED: Add inter-hand symmetry connections for coordinated movements
    for i in range(NUM_HAND):
        left_idx = NUM_POSE + i
        right_idx = NUM_POSE + NUM_HAND + i
        edges.add((left_idx, right_idx))
        edges.add((right_idx, left_idx))
    
    # 4. IMPROVED: Add face-to-hands connections (75-552 face landmarks)
    NOSE = 0
    LEFT_EYE = 2
    RIGHT_EYE = 5
    
    # Key face landmarks for ASL (mouth, eyes, eyebrows)
    FACE_KEY_POINTS = [75, 76, 77, 78, 79, 80]  # First few face landmarks
    
    # Connect face to hand centers for expressive signing
    LEFT_HAND_CENTER = NUM_POSE + 9   # Middle finger MCP
    RIGHT_HAND_CENTER = NUM_POSE + NUM_HAND + 9
    
    # Face-pose connections
    for face_point in [NOSE, LEFT_EYE, RIGHT_EYE]:
        edges.add((face_point, LEFT_HAND_CENTER))
        edges.add((LEFT_HAND_CENTER, face_point))
        edges.add((face_point, RIGHT_HAND_CENTER))
        edges.add((RIGHT_HAND_CENTER, face_point))
    
    # Face landmark internal connectivity (simplified)
    face_offset = NUM_POSE + 2 * NUM_HAND  # 75
    for i in range(0, NUM_FACE - 1, 10):  # Connect every 10th face landmark
        if i + 1 < NUM_FACE:
            edges.add((face_offset + i, face_offset + i + 1))
            edges.add((face_offset + i + 1, face_offset + i))
    
    # Convert to PyTorch Geometric format
    edge_list = list(edges)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    print(f"IMPROVED Graph: {TOTAL_NODES} nodes, {len(edge_list)} edges")
    print(f"Added face landmarks and enhanced connectivity for ASL recognition")
    
    return edge_index

# Data augmentation functions
def apply_spatial_augmentation(keypoints, angle_range=15, scale_range=0.1, translation_range=0.05):
    """Apply spatial augmentation to pose keypoints"""
    seq_len, num_nodes, num_features = keypoints.shape
    augmented = keypoints.copy()
    
    # Random rotation around z-axis
    angle = np.random.uniform(-angle_range, angle_range) * np.pi / 180
    cos_angle, sin_angle = np.cos(angle), np.sin(angle)
    
    # Random scale
    scale = np.random.uniform(1 - scale_range, 1 + scale_range)
    
    # Random translation
    tx = np.random.uniform(-translation_range, translation_range)
    ty = np.random.uniform(-translation_range, translation_range)
    
    for t in range(seq_len):
        # Apply rotation and scaling to x, y coordinates
        x_coords = augmented[t, :, 0]
        y_coords = augmented[t, :, 1]
        
        # Rotate
        new_x = x_coords * cos_angle - y_coords * sin_angle
        new_y = x_coords * sin_angle + y_coords * cos_angle
        
        # Scale and translate
        augmented[t, :, 0] = new_x * scale + tx
        augmented[t, :, 1] = new_y * scale + ty
        
        # Scale z coordinate
        augmented[t, :, 2] *= scale
    
    return augmented

def apply_temporal_augmentation(keypoints, speed_range=0.2):
    """Apply temporal augmentation (speed variation)"""
    seq_len = keypoints.shape[0]
    
    # Random speed factor
    speed_factor = np.random.uniform(1 - speed_range, 1 + speed_range)
    new_seq_len = int(seq_len * speed_factor)
    new_seq_len = max(10, min(new_seq_len, seq_len * 2))  # Reasonable bounds
    
    # Resample sequence
    old_indices = np.linspace(0, seq_len - 1, seq_len)
    new_indices = np.linspace(0, seq_len - 1, new_seq_len)
    
    augmented = np.zeros((new_seq_len, keypoints.shape[1], keypoints.shape[2]))
    
    for node in range(keypoints.shape[1]):
        for coord in range(keypoints.shape[2]):
            augmented[:, node, coord] = np.interp(new_indices, old_indices, keypoints[:, node, coord])
    
    return augmented

print("🚀 Improved normalization and preprocessing module loaded!")
print("📚 Based on successful TGCN implementations achieving 87.60% on WLASL-100")
print("✨ Features: spatial anchoring, temporal smoothing, improved graph connectivity")
print(f"📊 Supporting 553-node architecture: 33 pose + 42 hands + 478 face landmarks")
