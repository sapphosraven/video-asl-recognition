import os
import sys
import subprocess
import shutil
import json
import argparse

def install_dependencies():
    """Install required dependencies including OpenHands"""
    print("Installing dependencies...")
    
    dependencies = [
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "torch-geometric",
        "git+https://github.com/AI4Bharat/OpenHands.git@main#egg=OpenHands",
        "mediapipe>=0.8.9",
        "opencv-python>=4.5.0",
        "numpy>=1.19.0",
        "tqdm>=4.62.0"
    ]
    
    for dep in dependencies:
        try:
            print(f"Installing {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to install {dep}: {e}")
            continue
    
    print("Dependencies installation completed!")

def setup_wlasl300_class_mapping():
    """Create WLASL300 class mapping from the dataset"""
    print("Setting up WLASL300 class mapping...")
    
    # Load WLASL300 data
    wlasl300_path = os.path.join("pose_estimation", "data", "WLASL300.json")
    if not os.path.exists(wlasl300_path):
        print(f"WLASL300.json not found at {wlasl300_path}")
        return
    
    with open(wlasl300_path, 'r') as f:
        data = json.load(f)
    
    # Extract unique glosses and sort them
    glosses = sorted(list(set([item['gloss'] for item in data])))
    print(f"Found {len(glosses)} unique glosses in WLASL300")
    
    # Create class mapping (index to class)
    class_map = {str(i): gloss for i, gloss in enumerate(glosses)}
    
    # Save to wordlevelrecogntion directory
    os.makedirs("wordlevelrecogntion", exist_ok=True)
    with open("wordlevelrecogntion/class_map_wlasl300.json", 'w') as f:
        json.dump(class_map, f, indent=2)
    
    print(f"Saved WLASL300 class mapping with {len(class_map)} classes")

def download_openhands_model():
    """Download OpenHands WLASL300 model checkpoint"""
    print("Downloading OpenHands WLASL300 model...")
    
    try:
        subprocess.check_call([sys.executable, "download_wlasl300_model.py"])
        print("Model download completed!")
    except subprocess.CalledProcessError as e:
        print(f"Model download failed: {e}")
        print("You can download it manually later using download_wlasl300_model.py")

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
    parser = argparse.ArgumentParser(description="Setup WLASL dataset and OpenHands T-GCN models.")
    parser.add_argument("--skip-clone", action="store_true", help="Skip cloning WLASL repo if already cloned.")
    parser.add_argument("--skip-deps", action="store_true", help="Skip installing dependencies.")
    parser.add_argument("--skip-model", action="store_true", help="Skip downloading OpenHands model.")
    args = parser.parse_args()
    
    print("=== WLASL300 + OpenHands Setup ===")
    
    # Install dependencies
    if not args.skip_deps:
        install_dependencies()
    
    # Clone WLASL repo and prepare datasets
    if args.skip_clone:
        repo_dir = os.path.join("C:", "Raahima University", "Semester 6", "DL Project", "video-asl-recognition", "sentence_reconstruction", "data", "wlasl_repo")
        if not os.path.exists(repo_dir):
            print("Repository not found. Cannot skip cloning.")
            sys.exit(1)
    else:
        repo_dir = clone_wlasl_repo()
    
    prepare_dataset_subsets(repo_dir)
    
    # Setup WLASL300 class mapping
    setup_wlasl300_class_mapping()
    
    # Download OpenHands model
    if not args.skip_model:
        download_openhands_model()
    
    print("\n=== Setup Complete! ===")
    print("You can now run:")
    print("  python test_pipeline.py <video_path>  # Test the pipeline")
    print("  python app.py                         # Start web interface")
    print("\nFor WLASL300 T-GCN inference:")
    print("  - 300 ASL classes supported")
    print("  - Pose-based recognition with CNN fallback")
    print("  - MediaPipe keypoint extraction")

if __name__ == "__main__":
    main()
