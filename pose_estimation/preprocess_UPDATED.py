import os
import json
import cv2
import numpy as np
import shutil
from PIL import Image
from collections import defaultdict
import mediapipe as mp
import gc

# --- CONFIGURATION ---
# Set to True to extract all frames, or False to sample a fixed number
EXTRACT_ALL_FRAMES = True  # Set True for all frames, False for N frames
FRAMES_TO_SAMPLE = 16       # Used only if EXTRACT_ALL_FRAMES is False

VIDEOS_DIR = os.path.abspath(r'F:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\data\raw_videos_mp4')  # Directory with input videos
OUTPUT_DIR = os.path.abspath(r'F:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\data\processed')        # Output directory for processed data
JSON_PATH = os.path.abspath(r'F:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\data\WLASL100.json')    # Use WLASL100.json for this subset
OUTPUT_IMAGE_SIZE = (224, 224)

def load_json_data(json_path):
    """Load all classes/instances from WLASL100.json (no filtering)."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def extract_frames(video_path, extract_all=EXTRACT_ALL_FRAMES, frames_to_sample=FRAMES_TO_SAMPLE, start_frame=None, end_frame=None):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []
    # Restrict to sign segment if specified
    if start_frame is not None and end_frame is not None:
        start = max(0, int(start_frame))
        end = min(total_frames, int(end_frame))
        available_frames = end - start
        if available_frames <= 0:
            cap.release()
            return []
        if extract_all:
            frames_idx = np.arange(start, end)
        else:
            frames_idx = np.linspace(start, end - 1, min(frames_to_sample, available_frames), dtype=int)
    else:
        if extract_all:
            frames_idx = np.arange(0, total_frames)
        else:
            frames_idx = np.linspace(0, total_frames - 1, frames_to_sample, dtype=int)
    frames = []
    pose = mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    for idx in frames_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        # Try to center crop on person using MediaPipe Pose
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            left_shoulder = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
            center_x = int((left_shoulder.x + right_shoulder.x) / 2 * w)
            center_y = int((left_shoulder.y + right_shoulder.y) / 2 * h)
            min_side = min(h, w)
            half_side = min_side // 2
            x_min = max(0, center_x - half_side)
            x_max = min(w, center_x + half_side)
            y_min = max(0, center_y - half_side)
            y_max = min(h, center_y + half_side)
            # Adjust if crop goes out of bounds
            if x_max - x_min != min_side:
                if x_min == 0:
                    x_max = min_side
                else:
                    x_min = w - min_side
            if y_max - y_min != min_side:
                if y_min == 0:
                    y_max = min_side
                else:
                    y_min = h - min_side
            square_crop = frame[y_min:y_max, x_min:x_max]
            if square_crop.shape[0] != min_side or square_crop.shape[1] != min_side:
                # Fallback to center crop if something went wrong
                top = (h - min_side) // 2
                left = (w - min_side) // 2
                square_crop = frame[top:top+min_side, left:left+min_side]
        else:
            # Fallback: center crop
            min_side = min(h, w)
            top = (h - min_side) // 2
            left = (w - min_side) // 2
            square_crop = frame[top:top+min_side, left:left+min_side]
        frame_out = cv2.resize(square_crop, OUTPUT_IMAGE_SIZE)
        frames.append(frame_out)
    cap.release()
    return frames

def create_image_dataset(json_path=JSON_PATH,
                         videos_dir=VIDEOS_DIR,
                         output_dir=OUTPUT_DIR,
                         extract_all=EXTRACT_ALL_FRAMES,
                         frames_to_sample=FRAMES_TO_SAMPLE):
    print(f"Initialized. EXTRACT_ALL_FRAMES={extract_all}")
    data = load_json_data(json_path)
    os.makedirs(output_dir, exist_ok=True)
    splits = ['train', 'val', 'test']
    for split in splits:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
    # Create label mapping
    label_mapping = {entry['gloss']: idx for idx, entry in enumerate(data)}
    with open(os.path.join(output_dir, 'label_mapping.txt'), 'w') as f:
        for gloss, idx in sorted(label_mapping.items(), key=lambda x: x[1]):
            f.write(f'{gloss},{idx}\n')
    total_processed = 0
    total_failed = 0
    split_counts = {split: 0 for split in splits}
    class_counts = defaultdict(lambda: defaultdict(int))
    for entry in data:
        gloss = entry['gloss']
        label_idx = label_mapping[gloss]
        instances = entry['instances']
        print(f"\nProcessing class: {gloss} ({len(instances)} videos)")
        for split in splits:
            class_dir = os.path.join(output_dir, split, gloss)
            os.makedirs(class_dir, exist_ok=True)
        for instance in instances:
            video_id = instance['video_id']
            video_path = os.path.join(videos_dir, f'{video_id}.mp4')
            split = instance.get('split', 'train')
            # Use sign_start and sign_end if available
            sign_start = instance.get('sign_start', None)
            sign_end = instance.get('sign_end', None)
            if split not in splits:
                print(f"Warning: Unknown split '{split}' for {video_id}, defaulting to 'train'.")
                split = 'train'
            if not os.path.exists(video_path):
                print(f"Warning: Video not found - {video_path}")
                total_failed += 1
                continue
            try:
                frames = extract_frames(video_path, extract_all, frames_to_sample, sign_start, sign_end)
                if not frames:
                    print(f"No valid frames found in {video_path}")
                    total_failed += 1
                    continue
                for j, frame in enumerate(frames):
                    if not extract_all and frames_to_sample == 1:
                        save_path = os.path.join(output_dir, split, gloss, f'{video_id}.jpg')
                        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(save_path)
                        break
                    else:
                        save_path = os.path.join(output_dir, split, gloss, f'{video_id}_f{j}.jpg')
                        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(save_path)
                total_processed += 1
                split_counts[split] += 1
                class_counts[split][gloss] += 1
            except MemoryError:
                print(f"MemoryError processing {video_path}: skipping video.")
                total_failed += 1
                continue
            except Exception as e:
                print(f"Error processing {video_path}: {str(e)}")
                total_failed += 1
                continue
            finally:
                del frames
                gc.collect()
    print("\nProcessing completed!")
    print(f"Total videos processed successfully: {total_processed}")
    print(f"Total videos failed: {total_failed}")
    print("\nSplit distribution:")
    for split in splits:
        print(f"{split} set: {split_counts[split]} videos")
        print(f"Number of classes in {split}: {len(class_counts[split])}")

def cleanup_and_validate_dataset(output_dir='wlasl_dataset'):
    """
    Clean up empty folders and ensure consistency across splits.
    Returns statistics about class distribution.
    """
    splits = ['train', 'val', 'test']
    stats = {}
    classes_per_split = {}

    print("\nAnalyzing dataset structure...")

    # First pass: collect information about classes in each split
    for split in splits:
        split_path = os.path.join(output_dir, split)
        stats[split] = {}
        classes_per_split[split] = set()

        # Check each class folder
        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if not os.path.isdir(class_path):
                continue

            # Count images in the class
            images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            num_images = len(images)

            if num_images > 0:
                stats[split][class_name] = num_images
                classes_per_split[split].add(class_name)

    # Find classes present in all splits with non-empty folders
    valid_classes = set.intersection(*classes_per_split.values())

    # Find classes to remove (empty in any split or not present in all splits)
    all_classes = set.union(*classes_per_split.values())
    classes_to_remove = all_classes - valid_classes

    if classes_to_remove:
        print("\nRemoving inconsistent classes:")
        print("The following classes will be removed because they are either:")
        print("- Empty in one or more splits")
        print("- Not present in all splits")
        for class_name in sorted(classes_to_remove):
            print(f"- {class_name}")
            # Remove this class from all splits
            for split in splits:
                class_path = os.path.join(output_dir, split, class_name)
                if os.path.exists(class_path):
                    try:
                        shutil.rmtree(class_path)  # Remove directory and its contents
                        print(f"  Removed from {split}")
                    except OSError as e:
                        print(f"  Error removing from {split}: {e}")

    # Recalculate statistics after cleanup
    print("\nFinal dataset statistics:")
    for split in splits:
        split_path = os.path.join(output_dir, split)
        stats[split] = {}

        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if not os.path.isdir(class_path):
                continue

            images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            stats[split][class_name] = len(images)

        print(f"\n{split} set:")
        print(f"- Number of classes: {len(stats[split])}")
        print(f"- Total images: {sum(stats[split].values())}")

        # Print class distribution
        if stats[split]:
            class_counts = sorted(stats[split].items(), key=lambda x: x[1], reverse=True)
            print("\nClass distribution:")
            print("Top 5 largest classes:")
            for class_name, count in class_counts[:5]:
                print(f"  {class_name}: {count} images")
            print("Bottom 5 smallest classes:")
            for class_name, count in class_counts[-5:]:
                print(f"  {class_name}: {count} images")

    return stats


def suggest_fixes(stats):
    """
    Suggest fixes for dataset problems.
    """
    print("\nDataset Analysis and Suggestions:")

    # Check for class imbalance
    train_stats = stats['train']
    if train_stats:
        max_samples = max(train_stats.values())
        min_samples = min(train_stats.values())
        imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')

        print("\nClass Imbalance Analysis:")
        print(f"- Maximum samples in a class: {max_samples}")
        print(f"- Minimum samples in a class: {min_samples}")
        print(f"- Imbalance ratio: {imbalance_ratio:.2f}")

        if imbalance_ratio > 3:
            print("\nSuggestions for handling class imbalance:")
            print("1. Use WeightedRandomSampler in DataLoader:")
            print("""
    def create_weighted_sampler(dataset):
        class_counts = {}
        for path, _ in dataset.samples:
            label = os.path.basename(os.path.dirname(path))
            class_counts[label] = class_counts.get(label, 0) + 1

        weights = [1.0 / class_counts[os.path.basename(os.path.dirname(path))] 
                  for path, _ in dataset.samples]
        return WeightedRandomSampler(weights, len(weights))
            """)
            print("\n2. Consider data augmentation techniques:")
            print("   - Random rotation (±30 degrees)")
            print("   - Random brightness/contrast")
            print("   - Random horizontal flip (if appropriate)")

            print("\n3. Use weighted loss function:")
            print("""
    class_weights = torch.FloatTensor([1/count for count in class_counts.values()])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
            """)


def main():
    print("Starting preprocessing...")
    # First, make sure mediapipe is installed
    try:
        import mediapipe
    except ImportError:
        print("Installing required package: mediapipe")
        import subprocess
        subprocess.check_call(["pip", "install", "mediapipe"])

    # Step 1: Create dataset
    create_image_dataset()

    # Step 2: Clean up and validate dataset
    stats = cleanup_and_validate_dataset()

    # Step 3: Suggest fixes for any issues
    suggest_fixes(stats)

    print("\nDataset processing complete!")
    print("The dataset now contains:")
    print("1. Frames resized to 224x224")
    print("2. Consistent classes across all splits")
    print("3. No empty folders")


if __name__ == "__main__":
    main()