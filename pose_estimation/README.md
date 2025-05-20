# Pose Estimation for ASL Recognition

This module focuses on using pose estimation techniques to recognize American Sign Language (ASL) gestures from video inputs. We leverage the WLASL dataset and existing tools for efficient development.

## Directory Structure

- `src/` - Source code for the pose estimation pipeline

  - `utils/` - Utility functions
  - `models/` - Model definitions
  - `preprocessing/` - Data preprocessing scripts
  - `evaluation/` - Model evaluation code

- `data/` - Data directory (not tracked by git)
  - `raw/` - Raw video files
  - `processed/` - Processed keypoint data
  - `WLASL100/` - WLASL-100 dataset
  - `WLASL300/` - WLASL-300 dataset
  - `WLASL1000/` - WLASL-1000 dataset

## Getting Started

1. Run the setup scripts from the project root:

   ```
   python setup_directories.py
   python setup_wlasl.py
   ```

2. Download dataset videos using the script from WLASL:

   ```
   cd pose_estimation/src/preprocessing
   python video_downloader.py --dataset-dir ../../data/WLASL100 --split-json ../../data/WLASL100.json
   ```

3. Extract pose keypoints from videos:
   (Instructions will be added as the project develops)

## Implementation Notes

- We use MediaPipe for pose estimation as it provides lightweight but accurate hand and body keypoints
- Keypoint data is normalized and processed to create sequential data for model training
- Models are trained on WLASL-100, WLASL-300, and WLASL-1000 for comparison
