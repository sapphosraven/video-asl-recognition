"""
Script to compare two lists of video files and identify differences.
This helps team members identify which videos they are missing.
"""

import argparse
import os

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare two lists of video files')
    parser.add_argument('--file1', type=str, required=True, 
                      help='First text file with video filenames')
    parser.add_argument('--file2', type=str, required=True,
                      help='Second text file with video filenames')
    parser.add_argument('--output', type=str, default='video_comparison.txt',
                      help='Output text file (default: video_comparison.txt)')
    
    args = parser.parse_args()
    
    unique1, unique2 = compare_files(args.file1, args.file2, args.output)
    
    if unique1 == 0 and unique2 == 0:
        print("Both files contain the same set of videos!")
