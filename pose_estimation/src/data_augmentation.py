import numpy as np
import torch


class TimeseriesAugmenter:
    """
    Class for performing various data augmentations on time series data.
    These augmentations help reduce overfitting during model training.
    """
    
    @staticmethod
    def time_warp(x, sigma=0.2, knots=4):
        """
        Time warping augmentation.
        
        Args:
            x: Time series tensor of shape [batch, seq_len, features]
            sigma: Standard deviation of the noise
            knots: Number of anchor points for the warping
            
        Returns:
            Warped time series tensor of the same shape
        """
        try:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            if x.nelement() == 0:
                return x
            if not x.is_floating_point():
                x = x.float()
            orig_shape = x.shape
            if len(orig_shape) == 2:
                x = x.unsqueeze(0)
            batch_size, seq_len, features = x.shape
            if seq_len < 2:
                if len(orig_shape) == 2:
                    return x.squeeze(0)
                return x
            actual_knots = min(knots, seq_len // 2)
            anchor_points = np.linspace(0, 1, actual_knots+2)
            # Initialize warped time series
            x_warped = torch.zeros_like(x)
            for i in range(batch_size):
                f = np.cumsum(np.random.normal(loc=0, scale=sigma, size=actual_knots))
                f = np.concatenate([np.zeros(1), f, np.zeros(1)])  # Add endpoints
                if np.max(np.abs(f)) > 0:
                    f = f / np.max(np.abs(f))  # Normalize
                orig_points = np.linspace(0, 1, seq_len)
                # anchor_points and f are both (actual_knots+2,)
                warp_points = np.interp(orig_points, anchor_points, anchor_points + f)
                warp_idx = np.floor(warp_points * (seq_len-1)).astype(int)
                for j in range(seq_len):
                    if warp_idx[j] < seq_len:
                        x_warped[i, j] = x[i, warp_idx[j]]
            if len(orig_shape) == 2:
                x_warped = x_warped.squeeze(0)
            return x_warped
        except Exception:
            return x
    
    @staticmethod
    def magnitude_warp(x, sigma=0.2, knots=4):
        """
        Magnitude warping augmentation.
        
        Args:
            x: Time series tensor of shape [batch, seq_len, features]
            sigma: Standard deviation of the noise
            knots: Number of anchor points for the warping
            
        Returns:
            Magnitude warped time series tensor of the same shape
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
            
        orig_shape = x.shape
        
        # Make sure x is 3D
        if len(orig_shape) == 2:
            x = x.unsqueeze(0)  # Add batch dimension
            
        batch_size, seq_len, features = x.shape
        
        # Generate random warping
        x_warped = x.clone()
        
        for i in range(batch_size):
            for dim in range(features):
                # Generate random factors for each feature dimension
                f = np.cumsum(np.random.normal(loc=0, scale=sigma, size=knots))
                f = np.concatenate([np.zeros(1), f, np.zeros(1)])  # Add endpoints
                f = 1 + (f / np.max(np.abs(f)) * sigma)  # Scale factors
                
                # Apply warping by multiplying by factors
                warp_factors = np.interp(
                    np.linspace(0, 1, seq_len), 
                    np.linspace(0, 1, knots+2), 
                    f
                )
                x_warped[i, :, dim] = x[i, :, dim] * torch.tensor(warp_factors, dtype=x.dtype)
        
        # Restore original shape if necessary
        if len(orig_shape) == 2:
            x_warped = x_warped.squeeze(0)
            
        return x_warped
    
    @staticmethod
    def jitter(x, sigma=0.1):
        """
        Add random jitter noise to the time series data.
        
        Args:
            x: Time series tensor of shape [batch, seq_len, features] or [seq_len, features]
            sigma: Standard deviation of the noise
            
        Returns:
            Jittered time series tensor of the same shape
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        
        noise = torch.randn_like(x) * sigma
        return x + noise
    
    @staticmethod
    def scaling(x, sigma=0.1):
        """
        Apply random scaling to the time series data.
        
        Args:
            x: Time series tensor of shape [batch, seq_len, features] or [seq_len, features]
            sigma: Standard deviation of the scaling factor
            
        Returns:
            Scaled time series tensor of the same shape
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
            
        # Generate random scaling factors
        if len(x.shape) == 3:
            # [batch, seq_len, features]
            batch_size, _, features = x.shape
            factors = torch.randn(batch_size, 1, features) * sigma + 1
        else:
            # [seq_len, features]
            _, features = x.shape
            factors = torch.randn(1, features) * sigma + 1
            
        return x * factors
    
    @staticmethod
    def rotation(x):
        """
        Apply random rotation to the keypoints data.
        This assumes the keypoints are in xyz format (groups of 3 values).
        
        Args:
            x: Keypoints tensor of shape [batch, seq_len, features]
              where features are xyzxyz...
              
        Returns:
            Rotated keypoints tensor of the same shape
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
            
        orig_shape = x.shape
        
        # Make sure x is 3D
        if len(orig_shape) == 2:
            x = x.unsqueeze(0)  # Add batch dimension
            
        batch_size, seq_len, features = x.shape
        
        # Ensure features is divisible by 3
        if features % 3 != 0:
            print(f"Warning: Features dimension ({features}) is not divisible by 3. Skipping rotation.")
            return x
            
        num_keypoints = features // 3
        x_rotated = x.clone()
        
        for i in range(batch_size):
            # Generate random rotation matrix (simplified 2D rotation in XY plane)
            angle = np.random.uniform(-np.pi/6, np.pi/6)  # ±30 degrees
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            R = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
            
            # Apply rotation to each keypoint
            for j in range(num_keypoints):
                idx = j * 3
                for t in range(seq_len):
                    kp = x[i, t, idx:idx+3].numpy()
                    x_rotated[i, t, idx:idx+3] = torch.tensor(np.dot(R, kp), dtype=x.dtype)
        
        # Restore original shape if necessary
        if len(orig_shape) == 2:
            x_rotated = x_rotated.squeeze(0)
            
        return x_rotated
        
    @staticmethod
    def masking(x, mask_prob=0.1):
        """
        Randomly mask parts of the time series with zeros.
        
        Args:
            x: Time series tensor of shape [batch, seq_len, features] or [seq_len, features]
            mask_prob: Probability of masking a timestep
            
        Returns:
            Masked time series tensor of the same shape
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
            
        x_masked = x.clone()
        
        if len(x.shape) == 3:
            # [batch, seq_len, features]
            batch_size, seq_len, _ = x.shape
            mask = torch.rand(batch_size, seq_len, 1) < mask_prob
            x_masked[mask.expand_as(x)] = 0
        else:
            # [seq_len, features]
            seq_len, _ = x.shape
            mask = torch.rand(seq_len, 1) < mask_prob
            x_masked[mask.expand_as(x)] = 0
            
        return x_masked
    
    @staticmethod
    def permutation(x, max_segments=5):
        """
        Randomly permute segments of the time series.
        
        Args:
            x: Time series tensor of shape [batch, seq_len, features] or [seq_len, features]
            max_segments: Maximum number of segments to permute
            
        Returns:
            Permuted time series tensor of the same shape
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
            
        orig_shape = x.shape
        
        # Make sure x is 3D
        if len(orig_shape) == 2:
            x = x.unsqueeze(0)  # Add batch dimension
            
        batch_size, seq_len, features = x.shape
        x_permuted = x.clone()
        
        for i in range(batch_size):
            # Randomly choose the number of segments
            segments = np.random.randint(2, max_segments + 1)
            
            # Split the sequence into segments of roughly equal size
            segment_size = seq_len // segments
            indices = np.arange(segments)
            np.random.shuffle(indices)
            
            # Apply permutation
            for j in range(segments):
                if j < segments - 1:
                    orig_idx = j * segment_size
                    perm_idx = indices[j] * segment_size
                    x_permuted[i, perm_idx:perm_idx+segment_size] = x[i, orig_idx:orig_idx+segment_size]
                else:
                    # Last segment (might be longer due to integer division)
                    orig_idx = j * segment_size
                    perm_idx = indices[j] * segment_size
                    x_permuted[i, perm_idx:] = x[i, orig_idx:]
        
        # Restore original shape if necessary
        if len(orig_shape) == 2:
            x_permuted = x_permuted.squeeze(0)
            
        return x_permuted
    
    @staticmethod
    def time_shift(x, shift_ratio=0.2):
        """
        Shift the time series left or right by a random amount.
        
        Args:
            x: Time series tensor of shape [batch, seq_len, features] or [seq_len, features]
            shift_ratio: Maximum shift as a fraction of the sequence length
            
        Returns:
            Shifted time series tensor of the same shape
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
            
        orig_shape = x.shape
        
        # Make sure x is 3D
        if len(orig_shape) == 2:
            x = x.unsqueeze(0)  # Add batch dimension
            
        batch_size, seq_len, features = x.shape
        x_shifted = torch.zeros_like(x)
        
        for i in range(batch_size):
            # Random shift within [-shift_ratio, shift_ratio]
            shift = int(np.random.uniform(-shift_ratio, shift_ratio) * seq_len)
            
            # Apply shift
            if shift > 0:
                # Shift right
                x_shifted[i, shift:] = x[i, :seq_len-shift]
                x_shifted[i, :shift] = x[i, 0].unsqueeze(0).repeat(shift, 1)  # Repeat first frame
            elif shift < 0:
                # Shift left
                shift = abs(shift)
                x_shifted[i, :seq_len-shift] = x[i, shift:]
                x_shifted[i, seq_len-shift:] = x[i, -1].unsqueeze(0).repeat(shift, 1)  # Repeat last frame
            else:
                # No shift
                x_shifted[i] = x[i]
        
        # Restore original shape if necessary
        if len(orig_shape) == 2:
            x_shifted = x_shifted.squeeze(0)
            
        return x_shifted
    
    @staticmethod
    def apply_augmentations(x, augment_prob=0.5):
        """
        Apply a random combination of augmentations to the time series data.
        
        Args:
            x: Time series tensor of shape [batch, seq_len, features] or [seq_len, features]
            augment_prob: Probability of applying each augmentation
            
        Returns:
            Augmented time series tensor of the same shape
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
            
        # List of augmentation functions
        augmentations = [
            TimeseriesAugmenter.jitter,
            TimeseriesAugmenter.scaling,
            TimeseriesAugmenter.time_shift,
            TimeseriesAugmenter.masking,
            # More complex augmentations only applied with 0.5 probability
            lambda x: TimeseriesAugmenter.magnitude_warp(x) if np.random.random() < 0.5 else x,
            lambda x: TimeseriesAugmenter.time_warp(x) if np.random.random() < 0.5 else x,
            lambda x: TimeseriesAugmenter.permutation(x) if np.random.random() < 0.5 else x
        ]
        
        # Apply augmentations with probability
        x_aug = x.clone()
        for aug_fn in augmentations:
            if np.random.random() < augment_prob:
                x_aug = aug_fn(x_aug)
                
        return x_aug


# Example usage for testing
if __name__ == '__main__':
    # Create a sample batch of time series data
    batch_size = 2
    seq_len = 10
    features = 9  # Assuming 3 keypoints with (x,y,z) coordinates
    x = torch.randn(batch_size, seq_len, features)
    
    augmenter = TimeseriesAugmenter()
    
    # Apply individual augmentations
    x_jittered = augmenter.jitter(x)
    x_scaled = augmenter.scaling(x)
    x_rotated = augmenter.rotation(x)
    x_masked = augmenter.masking(x)
    x_permuted = augmenter.permutation(x)
    x_shifted = augmenter.time_shift(x)
    x_mag_warped = augmenter.magnitude_warp(x)
    x_time_warped = augmenter.time_warp(x)
    
    # Apply random combination of augmentations
    x_augmented = augmenter.apply_augmentations(x)
    
    print(f"Original shape: {x.shape}")
    print(f"Augmented shape: {x_augmented.shape}")
    
    # Check that shapes remain the same
    assert x.shape == x_augmented.shape, "Augmentation changed data shape!"
    print("All augmentations completed successfully.")
