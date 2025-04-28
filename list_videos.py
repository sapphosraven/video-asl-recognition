"""
Script to list all video files in a directory and save their names to a text file.
This helps with comparing available videos across team members.
"""

import os
import argparse
from pathlib import Path

def list_video_files(directory, output_file):
    """
    Lists all video files in the directory and writes their names to a text file.
    
    Args:
        directory: Path to the directory containing video files
        output_file: Path to save the list of video files
    """
    # Common video file extensions
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v']
    
    # Get absolute paths
    directory = os.path.abspath(directory)
    output_file = os.path.abspath(output_file)
    
    print(f"Scanning directory: {directory}")
    video_files = []
    
    # Walk through directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            if file_ext in video_extensions:
                # Get just the filename without path
                video_files.append(os.path.basename(file_path))
    
    # Sort alphabetically
    video_files.sort()
    
    # Write to output file
    with open(output_file, 'w') as f:
        for video in video_files:
            f.write(f"{video}\n")
    
    print(f"Found {len(video_files)} video files")
    print(f"Video list saved to {output_file}")
    
    return len(video_files)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='List all video files in a directory')
    parser.add_argument('--dir', type=str, default='pose_estimation/data/WLASL100',
                       help='Directory containing video files (default: pose_estimation/data/WLASL100)')
    parser.add_argument('--output', type=str, default='video_list.txt',
                       help='Output text file (default: video_list.txt)')
    
    args = parser.parse_args()
    
    count = list_video_files(args.dir, args.output)
    
    if count == 0:
        print("Warning: No video files found in the specified directory!")
