# Video ASL Recognition

## Overview

This project implements a deep learning-based American Sign Language (ASL) recognition system using video input. The system combines CNN-based image recognition with pose-based T-GCN models for enhanced accuracy using the WLASL300 dataset.

## Features

- **Video Segmentation**: Automatic segmentation of ASL videos into individual word clips using motion detection
- **Dual Recognition Models**:
  - CNN-based visual recognition for image features
  - T-GCN pose-based recognition using OpenHands framework with WLASL300 (300 classes)
- **Pose Estimation**: MediaPipe integration for real-time keypoint extraction
- **Sentence Reconstruction**: NLP post-processing to generate fluent sentences
- **Web Interface**: Flask-based UI for video upload and recognition

## Project Structure

- `app.py`: Flask web application for user interface
- `pipeline.py`: Main recognition pipeline with segmentation and dual inference
- `pose_inference.py`: T-GCN pose-based word recognition using OpenHands
- `wordlevelrecogntion/`: CNN-based word recognition module
- `sentence_reconstruction/`: NLP models for sentence reconstruction
- `pose_estimation/`: MediaPipe pose extraction and data processing
- `OpenHands/checkpoints/`: T-GCN model checkpoints for WLASL300

## WLASL300 Integration

The system now supports 300 ASL words from the WLASL300 dataset:

- **Pose-based T-GCN**: Primary inference method using temporal graph convolution
- **CNN Fallback**: Secondary method for robustness
- **Class Mapping**: 300 unique ASL signs with proper index mapping

## Getting Started

1. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Download WLASL300 T-GCN Model** (replace placeholder):

   ```bash
   # Download the actual pretrained model to replace the placeholder
   wget https://github.com/AI4Bharat/OpenHands/releases/download/v1.0.0/model_tgcn_wlasl300.pth -O OpenHands/checkpoints/model_tgcn_wlasl300.pth
   ```

3. **Test the Pipeline**:

   ```bash
   python test_pipeline.py path/to/your/asl_video.mp4
   ```

4. **Run Web Interface**:
   ```bash
   python app.py
   ```

## Technical Details

- **Pose Processing**: MediaPipe extracts 553 landmarks, reduced to 25 body joints
- **Sequence Processing**: Keypoint sequences padded/truncated to 60 frames
- **Model Architecture**: T-GCN with 25 joints, 3 channels (x,y,z), 300 output classes
- **Fallback Strategy**: If pose-based prediction fails or has low confidence, CNN model is used

## Technologies

- **OpenHands**: T-GCN models for pose-based ASL recognition
- **MediaPipe**: Real-time pose estimation and keypoint extraction
- **PyTorch**: Deep learning framework for model training and inference
- **Flask**: Web framework for user interface
- **OpenCV**: Video processing and computer vision
- **NumPy/SciPy**: Scientific computing and data processing
