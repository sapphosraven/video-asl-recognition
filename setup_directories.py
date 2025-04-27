import os
import sys

def create_directory_structure():
    """
    Creates the necessary directory structure for the pose estimation project
    """
    # Define base directory
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_estimation")
    
    # Define all needed directories
    directories = [
        # Main structure
        "",
        "src",
        "data",
        
        # Data subdirectories
        "data/raw",
        "data/processed",
        "data/WLASL100",
        "data/WLASL300",
        "data/WLASL1000",
        
        # Source code subdirectories
        "src/utils",
        "src/models",
        "src/preprocessing",
        "src/evaluation",
    ]
    
    # Create all directories
    for dir_path in directories:
        full_path = os.path.join(base_dir, dir_path)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            print(f"Created directory: {full_path}")
        else:
            print(f"Directory already exists: {full_path}")

if __name__ == "__main__":
    print("Setting up directory structure for ASL Recognition project...")
    create_directory_structure()
    print("Directory structure setup complete!")
