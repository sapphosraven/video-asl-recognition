import os
import sys
import subprocess
import shutil
import json
import argparse

def clone_wlasl_repo():
    """
    Clone the official WLASL repository
    """
    # Define target directory
    target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_estimation", "wlasl_repo")
    
    # Check if directory already exists
    if os.path.exists(target_dir):
        print(f"Repository directory {target_dir} already exists.")
        choice = input("Do you want to remove it and clone again? (y/n): ")
        if choice.lower() == 'y':
            shutil.rmtree(target_dir)
        else:
            print("Skipping cloning process.")
            return target_dir
    
    # Clone the repository
    print("Cloning WLASL repository...")
    try:
        subprocess.check_call(
            ["git", "clone", "https://github.com/dxli94/WLASL.git", target_dir],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"Successfully cloned WLASL repository to {target_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")
        sys.exit(1)
        
    return target_dir

def copy_necessary_files(repo_dir):
    """
    Copy necessary files from the cloned repo to our project structure
    """
    pose_est_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_estimation")
    
    # Files to copy with their source and destination paths relative to the repos
    files_to_copy = [
        # WLASL dataset JSON files
        (os.path.join(repo_dir, "start_kit", "WLASL_v0.3.json"), 
         os.path.join(pose_est_dir, "data", "WLASL_v0.3.json")),
        
        # Video preprocessing scripts
        (os.path.join(repo_dir, "start_kit", "video_downloader.py"), 
         os.path.join(pose_est_dir, "src", "preprocessing", "video_downloader.py")),
        
        # Any other necessary files from the repository
        # Add more as needed
    ]
    
    # Copy each file
    for src, dst in files_to_copy:
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied {src} to {dst}")
        else:
            print(f"Warning: Source file {src} not found")

def setup_dataset_subsets():
    """
    Set up the WLASL100, WLASL300, and WLASL1000 dataset splits
    """
    pose_est_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_estimation")
    json_file = os.path.join(pose_est_dir, "data", "WLASL_v0.3.json")
    
    if not os.path.exists(json_file):
        print(f"Error: WLASL dataset JSON file not found at {json_file}")
        return
    
    try:
        # Load the JSON data
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Create subset JSONs
        subsets = {
            'WLASL100': data[:100],
            'WLASL300': data[:300], 
            'WLASL1000': data[:1000]
        }
        
        # Save each subset
        for name, subset in subsets.items():
            output_file = os.path.join(pose_est_dir, "data", f"{name}.json")
            with open(output_file, 'w') as f:
                json.dump(subset, f, indent=2)
            print(f"Created {name} subset with {len(subset)} classes at {output_file}")
            
    except Exception as e:
        print(f"Error processing dataset: {e}")

def main():
    parser = argparse.ArgumentParser(description='Setup WLASL dataset for ASL Recognition project')
    parser.add_argument('--skip-clone', action='store_true', help='Skip cloning the repository')
    args = parser.parse_args()
    
    if not args.skip_clone:
        repo_dir = clone_wlasl_repo()
        copy_necessary_files(repo_dir)
    else:
        repo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_estimation", "wlasl_repo")
        if not os.path.exists(repo_dir):
            print(f"Error: Repository directory {repo_dir} does not exist. Cannot skip cloning.")
            sys.exit(1)
    
    setup_dataset_subsets()
    print("WLASL dataset setup complete!")

if __name__ == "__main__":
    main()
