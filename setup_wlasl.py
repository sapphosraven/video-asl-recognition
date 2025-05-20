import os
import sys
import subprocess
import shutil
import json
import argparse

def clone_wlasl_repo():
    """
    Clone the WLASL repository into 'data/wlasl_repo'
    """
    target_dir = os.path.join("C:", "Raahima University", "Semester 6", "DL Project", "video-asl-recognition", "sentence_reconstruction", "data", "wlasl_repo")
    
    if os.path.exists(target_dir):
        print(f"Repository already exists at {target_dir}. Skipping clone.")
        return target_dir
    
    print("Cloning WLASL repository...")
    try:
        subprocess.check_call(["git", "clone", "https://github.com/dxli94/WLASL.git", target_dir])
        print(f"Cloned into {target_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")
        sys.exit(1)
    
    return target_dir

def prepare_dataset_subsets(repo_dir):
    """
    Create WLASL100, WLASL300, WLASL1000 JSON files inside data/
    """
    json_path = os.path.join(repo_dir, "start_kit", "WLASL_v0.3.json")
    
    if not os.path.exists(json_path):
        print(f"Dataset JSON not found at {json_path}")
        sys.exit(1)
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    subsets = {
        "WLASL100.json": data[:100],
        "WLASL300.json": data[:300],
        "WLASL1000.json": data[:1000],
    }
    
    data_dir = os.path.join("C:", "Raahima University", "Semester 6", "DL Project", "video-asl-recognition", "sentence_reconstruction", "data")
    os.makedirs(data_dir, exist_ok=True)
    
    for filename, subset in subsets.items():
        output_path = os.path.join(data_dir, filename)
        with open(output_path, "w") as f:
            json.dump(subset, f, indent=2)
        print(f"Saved {filename} with {len(subset)} entries.")

def main():
    parser = argparse.ArgumentParser(description="Prepare WLASL dataset subsets.")
    parser.add_argument("--skip-clone", action="store_true", help="Skip cloning WLASL repo if already cloned.")
    args = parser.parse_args()
    
    if args.skip_clone:
        repo_dir = os.path.join("C:", "Raahima University", "Semester 6", "DL Project", "video-asl-recognition", "sentence_reconstruction", "data", "wlasl_repo")
        if not os.path.exists(repo_dir):
            print("Repository not found. Cannot skip cloning.")
            sys.exit(1)
    else:
        repo_dir = clone_wlasl_repo()
    
    prepare_dataset_subsets(repo_dir)
    print("Dataset preparation complete!")

if __name__ == "__main__":
    main()
