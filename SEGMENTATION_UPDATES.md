# ASL Recognition - Enhanced Segmentation

## Improvements to the Video Segmentation Pipeline

This update replaces our simple motion-based word segmentation with a more robust approach based on techniques from the WLASL Recognition and Translation project.

### What Changed

1. **Enhanced Motion Detection**:

   - Replaced basic frame differencing with a center-weighted approach that focuses on hand regions
   - Implemented adaptive thresholding that adjusts based on video content
   - Added temporal smoothing to reduce jitter and false positives

2. **Improved Segment Boundaries**:

   - Detects periods of significant motion followed by stillness
   - Uses minimum segment length to avoid over-segmentation
   - Handles edge cases like continuous motion or very little motion

3. **Naming Convention**:

   - Old: `temp_clip_X.mp4` (where X is a sequential number)
   - New: `videoname_start_end.mp4` (where start and end are frame indices)
   - This format matches WLASL conventions and provides more information

4. **Comprehensive Pipeline**:
   - Added a unified `process_asl_video()` function that handles the entire pipeline
   - Integrated detailed logging to monitor performance
   - Added timing measurements for each step

### Using the New Pipeline

The API remains the same, so no code changes are needed if you're already using our pipeline functions:

```python
# Import the main function
from pipeline import process_asl_video

# Process a video end-to-end
results = process_asl_video("path/to/video.mp4")

# Access results
clips = results['clips']        # List of clip file paths
words = results['words']        # List of recognized words
sentence = results['sentence']  # Reconstructed sentence
timing = results['timing']      # Processing time details
```

For direct CLI usage, you can also run:

```
python pipeline.py path/to/video.mp4
```

### Web Interface Updates

The web interface now shows:

- Each detected segment as a playable video clip
- Timestamps for each segment
- Recognition results aligned with clips
- Total processing time

### Performance Notes

The enhanced segmentation is more computationally intensive than our previous approach but delivers more accurate word boundaries, especially for:

- Continuous signing with minimal pauses
- Signers with different styles and speeds
- Videos with varying background and lighting conditions

The algorithm adapts to the frame rate of the input video to ensure consistent results across different recording conditions.
