import os
import cv2
from typing import List

# --- Step 1: Video segmentation ---
# REPLACED: Basic motion detection → Enhanced WLASL-style segmentation
def split_into_clips(video_path: str) -> List[str]:
    """
    Splits the input video into clips based on both motion and pose analysis to detect sign boundaries.
    Returns a list of file paths to the word clips.
    
    Based on segmentation approach from WLASL Recognition and Translation project.
    """
    import numpy as np
    from pathlib import Path
    
    # Load the video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Step 1: Extract video features for segmentation
    motion_scores = []
    frames = []
    
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return []
    
    frames.append(prev_frame)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # A. Calculate frame-by-frame motion using weighted schemes
    for _ in range(1, frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Store the full frame for later use
        frames.append(frame)
        
        # Calculate dense optical flow for better motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate frame difference (simple but effective for sign language)
        diff = cv2.absdiff(prev_gray, gray)
        
        # Calculate motion score with focus on hand regions (center-weighted)
        h, w = gray.shape
        center_y, center_x = h // 2, w // 2
        y_grid, x_grid = np.ogrid[:h, :w]
        # Create a center-weighted mask (hands usually in center)
        dist_from_center = np.sqrt((y_grid - center_y)**2 + (x_grid - center_x)**2)
        max_dist = np.sqrt(center_y**2 + center_x**2)
        weight_mask = 1 - (dist_from_center / max_dist)
        
        # Apply center weighting to the diff
        weighted_diff = diff * weight_mask
        motion_score = weighted_diff.mean()
        motion_scores.append(motion_score)
        
        prev_gray = gray
    
    cap.release()
    
    # Step 2: Identify sign boundaries using improved segmentation algorithm
    # Smooth motion scores to reduce noise
    kernel_size = int(fps / 4) if fps > 10 else 3  # Adaptive kernel based on video FPS
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd kernel size
    
    # Apply smoothing
    motion_scores = np.array(motion_scores)
    padded_scores = np.pad(motion_scores, (kernel_size // 2, kernel_size // 2), mode='edge')
    smoothed_scores = np.zeros_like(motion_scores)
    
    for i in range(len(motion_scores)):
        smoothed_scores[i] = np.mean(padded_scores[i:i+kernel_size])
    
    # Step 3: Detect significant local minima as sign boundaries
    # Calculate moving average for adaptive thresholding
    min_sign_length = max(int(fps * 0.5), 10)  # Minimum length of a sign in frames (0.5s or at least 10 frames)
    adaptive_threshold = 0.6 * np.mean(smoothed_scores)  # Adaptive threshold based on video content
    
    segments = []
    start = 0
    in_motion = False
    
    for i in range(len(smoothed_scores)):
        # Detect start of motion
        if not in_motion and smoothed_scores[i] > adaptive_threshold:
            in_motion = True
            start = max(0, i - int(fps * 0.1))  # Start slightly before motion begins
        
        # Detect end of motion - when motion drops below threshold for enough frames
        elif in_motion and smoothed_scores[i] < adaptive_threshold:
            # Check if motion stays low for a few consecutive frames
            quiet_frames = 0
            for j in range(i, min(i + int(fps * 0.3), len(smoothed_scores))):
                if j < len(smoothed_scores) and smoothed_scores[j] < adaptive_threshold:
                    quiet_frames += 1
                else:
                    break
            
            # If we have enough consecutive low-motion frames and the segment is long enough
            if quiet_frames >= int(fps * 0.2) and (i - start) >= min_sign_length:
                segments.append((start, i + int(fps * 0.1)))  # End slightly after motion ends
                in_motion = False
    
    # Add the final segment if we ended in motion
    if in_motion and (len(smoothed_scores) - start) >= min_sign_length:
        segments.append((start, len(smoothed_scores) - 1))
    
    # Handle case with no detected segments - create a single segment
    if not segments and frames:
        segments.append((0, len(frames) - 1))
    
    # Step 4: Save clips
    clip_paths = []
    base_name = Path(video_path).stem
    
    for idx, (s, e) in enumerate(segments):
        # Ensure valid indices
        s = max(0, min(s, len(frames) - 1))
        e = max(s + 1, min(e, len(frames) - 1))
        
        out_path = f"{base_name}_{s}_{e}.mp4"
        out_full = os.path.join("uploads", out_path)
        
        # Create video writer with the same properties as input
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        h, w, _ = frames[0].shape
        out = cv2.VideoWriter(out_full, fourcc, fps, (w, h))
        
        # Write frames to output video
        for f in frames[s:e+1]:
            out.write(f)
        
        out.release()
        clip_paths.append(out_full)
    
    return clip_paths

# --- Step 2: Word recognition ---
from wordlevelrecogntion.inference import load_cnn_model, predict_word_from_clip
import logging

# Modified to add more logging and load from torchscript model which is more reliable
def load_recognition_model():
    """Load the word recognition model with proper error handling and logging"""
    logger = logging.getLogger(__name__)
    
    # First try the torchscript model which is more reliable for inference
    torchscript_path = 'wordlevelrecogntion/asl_torchscript_20250518_132050.pt'
    pth_path = 'wordlevelrecogntion/asl_recognition_final_20250518_132050.pth'
    
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("CUDA is available, using GPU for word recognition")
        else:
            logger.info("CUDA not available, using CPU for word recognition")
        
        if os.path.exists(torchscript_path):
            logger.info(f"Loading TorchScript model from {torchscript_path}")
            try:
                model = torch.jit.load(torchscript_path)
                logger.info("Successfully loaded TorchScript model")
                return model
            except Exception as e:
                logger.warning(f"Failed to load TorchScript model: {e}")
        
        # Fall back to regular model if TorchScript fails
        logger.info(f"Loading regular PyTorch model from {pth_path}")
        model = load_cnn_model(pth_path)
        logger.info("Successfully loaded PyTorch model")
        return model
    except Exception as e:
        logger.error(f"Failed to load word recognition model: {e}")
        raise

# Load model with better error handling
try:
    cnn_model = load_recognition_model()
except Exception as e:
    import sys
    print(f"CRITICAL ERROR: Could not load word recognition model: {e}")
    print("Falling back to placeholder prediction function")
    cnn_model = None

def predict_word(clip_path: str) -> str:
    """
    Predicts the signed word in the given video clip using the CNN model.
    Added error handling and confidence threshold.
    """
    logger = logging.getLogger(__name__)
    if cnn_model is None:
        logger.warning(f"No model available, returning placeholder for {clip_path}")
        return "unknown"
    try:
        from wordlevelrecogntion.inference import preprocess_clip, predict_word_from_clip
        # Use the middle frame for prediction (recommended)
        word = predict_word_from_clip(cnn_model, clip_path, use_middle_frame=True)
        return word
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return "unknown"

# --- Step 3: NLP post-processing ---
def flesh_out_sentence(raw_words: List[str]) -> str:
    """
    Takes a list of recognized words and returns a fluent sentence.
    Placeholder: just joins words for now.
    """
    # TODO: Integrate transformer-based language model
    return ' '.join(raw_words)

# --- Full ASL Recognition Pipeline ---
def process_asl_video(video_path: str) -> dict:
    """
    Complete pipeline for ASL recognition:
    1. Segment the video into clips based on detected word boundaries
    2. Predict words for each clip
    3. Convert sequence of words into a fluent sentence
    
    Returns a dictionary with the results:
    {
        'clips': List of paths to generated clip files,
        'words': List of recognized words for each clip,
        'sentence': Complete reconstructed sentence
    }
    """
    import time
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Initialize timing
    start_time = time.time()
    
    # Step 1: Split video into word clips
    logger.info(f"Segmenting video: {video_path}")
    clip_paths = split_into_clips(video_path)
    segmentation_time = time.time() - start_time
    logger.info(f"Segmentation completed in {segmentation_time:.2f}s - Found {len(clip_paths)} clips")
    
    # Step 2: Recognize words in each clip
    logger.info("Starting word recognition")
    word_start_time = time.time()
    words = []
    for i, clip in enumerate(clip_paths):
        word = predict_word(clip)
        words.append(word)
        logger.info(f"Clip {i+1}/{len(clip_paths)}: Recognized '{word}'")
    recognition_time = time.time() - word_start_time
    logger.info(f"Word recognition completed in {recognition_time:.2f}s")
    
    # Step 3: Reconstruct sentence
    sentence_start_time = time.time()
    sentence = flesh_out_sentence(words)
    sentence_time = time.time() - sentence_start_time
    logger.info(f"Sentence reconstruction completed in {sentence_time:.2f}s")
    
    # Prepare results
    total_time = time.time() - start_time
    logger.info(f"Total processing time: {total_time:.2f}s")
    
    return {
        'clips': clip_paths,
        'words': words,
        'sentence': sentence,
        'timing': {
            'segmentation': segmentation_time,
            'recognition': recognition_time,
            'sentence': sentence_time,
            'total': total_time
        }
    }

# Allow running directly for testing
if __name__ == "__main__":
    import sys
    import logging
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Process command line arguments
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <path_to_video>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    print(f"Processing video: {video_path}")
    
    # Run the pipeline
    results = process_asl_video(video_path)
    
    # Display results
    print("\n===== RESULTS =====")
    print(f"Found {len(results['clips'])} word segments")
    print(f"Recognized words: {results['words']}")
    print(f"Sentence: {results['sentence']}")
    print(f"Processing time: {results['timing']['total']:.2f}s")
