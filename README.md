# Video ASL Recognition

## Overview

This project implements a deep learning-based American Sign Language (ASL) recognition system using video input. The system utilizes pose estimation techniques to track hand and body movements for recognizing sign language gestures.

## Project Structure

- `testcuda.py`: Script to verify CUDA availability for GPU acceleration
- `pipeline.txt`: Detailed project roadmap and implementation plan
- `requirements.txt`: List of dependencies required for the project

## Features (Planned)

- Real-time pose estimation for ASL recognition
- Multi-modal approach combining pose keypoints and visual features
- Support for common ASL signs and phrases
- User-friendly interface for testing and demonstration

## Getting Started

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run `python testcuda.py` to verify GPU support

## Technologies

- MediaPipe/OpenPose for pose estimation
- PyTorch/TensorFlow for deep learning models
- Python for implementation and data processing
