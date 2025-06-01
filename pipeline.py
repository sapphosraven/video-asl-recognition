import os
import cv2
import torch
import numpy as np
from typing import List
import logging
from pose_inference import load_pose_model, predict_word_from_pose
from wordlevelrecogntion.inference import load_cnn_model, predict_word_from_clip

# Configure logger
logger = logging.getLogger(__name__)

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pose_model = load_pose_model(device=device)
cnn_model = load_cnn_model('wordlevelrecogntion/asl_recognition_final_20250518_132050.pth')

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

def predict_word(clip_path: str) -> str:
    logger = logging.getLogger(__name__)
    clip_keypoints_path = clip_path.replace(".mp4", "_keypoints.npz")
    if os.path.exists(clip_keypoints_path):
        try:
            arr = np.load(clip_keypoints_path)["arr_0"]
            word = predict_word_from_pose(pose_model, arr, device=device)
            return word
        except Exception as e:
            logger.warning(f"Pose-based inference failed: {e}. Falling back to CNN.")
    # Fallback to CNN-based prediction
    try:
        word = predict_word_from_clip(cnn_model, clip_path, min_confidence=0.25)
        return word
    except Exception as e:
        logger.error(f"Error in CNN prediction: {e}")
        return "unknown"

# --- Step 3: Sentence reconstruction ---
def flesh_out_sentence(sign_clips: List[str], pause_duration: int = 500):
    """
    Given a list of sign clips, reconstruct the sentence by inserting pauses
    where necessary based on the analysis of the sign clips.
    """
    import moviepy.editor as mpy
    
    # Load the sign clips
    clips = [mpy.VideoFileClip(c) for c in sign_clips]
    
    # Calculate durations and decide pauses
    final_clips = []
    for i, clip in enumerate(clips):
        final_clips.append(clip)
        
        # Add a pause between signs if not the last sign
        if i < len(clips) - 1:
            pause = mpy.ColorClip(size=clip.size, color=(255,255,255), duration=pause_duration/1000)
            final_clips.append(pause)
    
    # Concatenate all clips
    final_video = mpy.concatenate_videoclips(final_clips, method="compose")
    
    return final_video

def process_asl_video(video_path: str):
    """
    Main processing function for the ASL video.
    1. Splits the video into clips containing individual signs.
    2. For each clip, it predicts the corresponding word in ASL.
    3. Reconstructs the sentence by combining the sign clips with appropriate pauses.
    """
    # Step 1: Video segmentation
    clip_paths = split_into_clips(video_path)
    
    if not clip_paths:
        logger.warning("No clips found for the given video.")
        return None
    
    # Step 2: Word prediction for each clip
    words = []
    for clip in clip_paths:
        word = predict_word(clip)
        words.append(word)
        logger.info(f"Predicted word for {clip}: {word}")
    
    # Step 3: Sentence reconstruction
    # For now, we just concatenate the words with a space
    sentence = " ".join(words)
    logger.info(f"Reconstructed sentence: {sentence}")
    
    # TODO: Improve sentence reconstruction with proper timing and pauses
    final_video = flesh_out_sentence(clip_paths)
    
    return final_video