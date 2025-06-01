"""
Test script for the full ASL recognition pipeline with improved word recognition
and WLASL300 T-GCN pose-based inference
"""
import os
import sys
import logging
import numpy as np
from pipeline import process_asl_video
from pose_inference import load_pose_model, predict_word_from_pose
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_pose_inference_on_sample_clip():
    """Test pose-based inference with dummy data"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = "OpenHands/checkpoints/model_tgcn_wlasl300.pth"
        
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found at {checkpoint_path}, skipping pose test")
            return
        
        # Load model
        pose_model = load_pose_model(checkpoint_path, device=device)
        logger.info("Pose model loaded successfully for testing")
        
        # Create dummy keypoints for testing (30 frames, 25 joints, 3 coordinates)
        dummy_keypoints = np.random.rand(30, 25, 3).astype(np.float32)
        
        # Test prediction
        predicted_word = predict_word_from_pose(pose_model, dummy_keypoints, device=device)
        logger.info(f"Pose inference test completed. Predicted word: '{predicted_word}'")
        
        assert isinstance(predicted_word, str), "Prediction should return a string"
        logger.info("✓ Pose inference test passed")
        
    except Exception as e:
        logger.error(f"Pose inference test failed: {e}")
        logger.warning("Continuing with other tests...")

def test_pose_inference_on_dummy():
    """Test pose inference on dummy keypoints"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pose_model = load_pose_model(device=device)
    
    # Create dummy keypoints (30 frames, 25 joints, 3 coordinates)
    dummy_keypoints = np.random.rand(30, 25, 3).astype(np.float32)
    np.savez("dummy_keypoints.npz", arr_0=dummy_keypoints)
    
    # Load dummy keypoints
    arr = np.load("dummy_keypoints.npz")["arr_0"]
    
    # Predict word from dummy keypoints
    word = predict_word_from_pose(pose_model, arr, device=device)
    
    assert isinstance(word, str), "Prediction should return a string"
    print(f"Dummy pose inference returned: {word}")

def test_on_video(video_path):
    """Run the full pipeline on a video and display results"""
    print(f"\n{'='*50}")
    print(f"TESTING PIPELINE ON: {os.path.basename(video_path)}")
    print(f"{'='*50}")
    
    # Process the video
    results = process_asl_video(video_path)
    
    # Display results
    print(f"\nFOUND {len(results['clips'])} SEGMENTS:")
    for i, (clip, word) in enumerate(zip(results['clips'], results['words'])):
        clip_name = os.path.basename(clip)
        print(f"  {i+1}. {clip_name} → '{word}'")
    
    print(f"\nSENTENCE: {results['sentence']}")
    print(f"\nTIMING: {results['timing']['total']:.2f}s total")
    print(f"  - Segmentation: {results['timing']['segmentation']:.2f}s")
    print(f"  - Word recognition: {results['timing']['recognition']:.2f}s")
    
    return results

if __name__ == "__main__":
    # Test pose inference on dummy data
    print("Testing pose inference on dummy data...")
    test_pose_inference_on_dummy()
    
    # First test pose inference if available
    print("Testing pose-based inference...")
    test_pose_inference_on_sample_clip()
    
    if len(sys.argv) < 2:
        print("Usage: python test_pipeline.py <path_to_video> [path_to_another_video] ...")
        sys.exit(1)
    
    # Test on all videos provided
    for video_path in sys.argv[1:]:
        test_on_video(video_path)
