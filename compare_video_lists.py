"""
Script to compare two lists of video files and identify differences.
This helps team members identify which videos they are missing.
"""

import os
import json

def read_file_lines(filename):
    """Read lines from a file and return as a set."""
    with open(filename, 'r') as f:
        return set(line.strip() for line in f if line.strip())

def compare_files(file1, file2, output_file):
    """
    Compare two text files line by line and identify differences.
    
    Args:
        file1: First text file with video filenames
        file2: Second text file with video filenames
        output_file: Output text file to save differences
    """
    print(f"Comparing {file1} and {file2}...")
    
    # Read files into sets
    files1 = read_file_lines(file1)
    files2 = read_file_lines(file2)
    
    # Get unique elements in each set
    only_in_file1 = files1 - files2
    only_in_file2 = files2 - files1
    
    # Write results to output file
    with open(output_file, 'w') as f:
        if only_in_file1:
            f.write(f"Files unique to {os.path.basename(file1)}:\n")
            f.write("="*50 + "\n")
            for item in sorted(only_in_file1):
                f.write(f"{item}\n")
            f.write("\n\n")
        
        if only_in_file2:
            f.write(f"Files unique to {os.path.basename(file2)}:\n")
            f.write("="*50 + "\n")
            for item in sorted(only_in_file2):
                f.write(f"{item}\n")
    
    # Print summary
    print(f"Found {len(only_in_file1)} files unique to {os.path.basename(file1)}")
    print(f"Found {len(only_in_file2)} files unique to {os.path.basename(file2)}")
    print(f"Comparison results saved to {output_file}")
    
    return len(only_in_file1), len(only_in_file2)

# Hardcoded paths
video_dir = r"f:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\data\WLASL100"
json_path = r"f:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\data\WLASL100.json"

# Get all video filenames (without extension) in the directory
video_files = set(os.path.splitext(f)[0] for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f)))

# Load all video_ids from the JSON
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

json_video_ids = set()
for entry in data:
    for inst in entry.get("instances", []):
        json_video_ids.add(str(inst["video_id"]))

# Find files in the directory not referenced in the JSON
missing_in_json = [f for f in video_files if f not in json_video_ids]

print("Files in directory but missing from JSON:")
for fname in missing_in_json:
    print(fname)
print(f"Total missing: {len(missing_in_json)}")
