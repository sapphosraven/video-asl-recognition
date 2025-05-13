import os
import json
import cv2
import numpy as np
from PIL import Image
from collections import defaultdict
import mediapipe as mp
import shutil


class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        )

    def detect_and_crop_hands(self, frame, confidence_threshold=0.5):
        """
        Detect hands in the frame and return a cropped image containing both hands
        with some padding around them.
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]

        # Detect hands
        results = self.hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return None, 0.0  # No hands detected

        # Calculate average confidence
        confidence = 0.0
        num_hands = 0
        if results.multi_handedness:
            for hand_score in results.multi_handedness:
                confidence += hand_score.classification[0].score
                num_hands += 1
            confidence = confidence / num_hands

            if confidence < confidence_threshold:
                return None, confidence

        # Get bounding box for all detected hands
        x_min, y_min = width, height
        x_max, y_max = 0, 0

        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * width), int(landmark.y * height)
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)

        # Add padding around hands (30% of hand size)
        padding_x = int((x_max - x_min) * 0.3)
        padding_y = int((y_max - y_min) * 0.3)

        x_min = max(0, x_min - padding_x)
        x_max = min(width, x_max + padding_x)
        y_min = max(0, y_min - padding_y)
        y_max = min(height, y_max + padding_y)

        # Crop the image to contain just the hands with padding
        hands_crop = frame[y_min:y_max, x_min:x_max]

        return (hands_crop, confidence) if hands_crop.size > 0 else (None, confidence)


def load_json_data(json_path, top_n=100):
    """Load JSON and filter for top N most common classes"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Count instances per class
    class_counts = defaultdict(int)
    for entry in data:
        class_counts[entry['gloss']] += len(entry['instances'])

    # Get top N classes
    top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_class_names = {item[0] for item in top_classes}

    # Filter data for top classes only
    filtered_data = [entry for entry in data if entry['gloss'] in top_class_names]

    total_instances = sum(len(entry['instances']) for entry in filtered_data)
    print(f"Found {total_instances} videos in top {top_n} classes")

    return filtered_data


def extract_best_frame(video_path, hand_detector, frames_to_sample=5):
    """
    Extract multiple frames from the video and return the best one
    with most confident hand detection
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames == 0 or fps == 0:
        cap.release()
        return None, 0.0

    # Calculate frame indices to sample
    # Skip first and last 15% of frames to avoid intro/outro
    start_frame = int(total_frames * 0.15)
    end_frame = int(total_frames * 0.85)
    frames_to_check = np.linspace(start_frame, end_frame, frames_to_sample, dtype=int)

    best_frame = None
    best_confidence = 0.0

    for frame_idx in frames_to_check:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        # Detect hands and get confidence
        hands_crop, confidence = hand_detector.detect_and_crop_hands(frame)

        # Update best frame if this one has higher confidence
        if confidence > best_confidence:
            best_confidence = confidence
            best_frame = frame

    cap.release()
    return best_frame, best_confidence


def create_image_dataset(json_path='WLASL_v0.3.json',
                         videos_dir='videos',
                         output_dir='wlasl_dataset',
                         train_split=0.8,
                         val_split=0.1,
                         test_split=0.1,
                         frames_to_sample=5):
    # Initialize hand detector
    hand_detector = HandDetector()
    print("Initialized hand detector...")

    # Load and filter data for top 100 classes
    data = load_json_data(json_path, top_n=100)

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    splits = ['train', 'val', 'test']
    for split in splits:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

    # Create label mapping
    label_mapping = {entry['gloss']: idx for idx, entry in enumerate(data)}

    # Save label mapping
    with open(os.path.join(output_dir, 'label_mapping.txt'), 'w') as f:
        for gloss, idx in sorted(label_mapping.items(), key=lambda x: x[1]):
            f.write(f'{gloss},{idx}\n')

    # Counters for tracking
    total_processed = 0
    total_failed = 0
    no_hands_detected = 0
    split_counts = {split: 0 for split in splits}
    class_counts = defaultdict(lambda: defaultdict(int))

    # Process each class
    for entry in data:
        gloss = entry['gloss']
        label_idx = label_mapping[gloss]
        instances = entry['instances']

        print(f"\nProcessing class: {gloss} ({len(instances)} videos)")

        # Create class directories in each split
        for split in splits:
            class_dir = os.path.join(output_dir, split, gloss)
            os.makedirs(class_dir, exist_ok=True)

        # Determine splits
        n_total = len(instances)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)

        # Shuffle instances
        np.random.shuffle(instances)

        # Process each video
        for i, instance in enumerate(instances):
            video_id = instance['video_id']
            video_path = os.path.join(videos_dir, f'{video_id}.mp4')

            if not os.path.exists(video_path):
                print(f"Warning: Video not found - {video_path}")
                total_failed += 1
                continue

            # Determine split for this instance
            if i < n_train:
                split = 'train'
            elif i < n_train + n_val:
                split = 'val'
            else:
                split = 'test'

            try:
                # Extract best frame with hands
                best_frame, confidence = extract_best_frame(video_path, hand_detector, frames_to_sample)

                if best_frame is None:
                    print(f"No suitable frame found in {video_path}")
                    no_hands_detected += 1
                    continue

                # Detect and crop hands from the best frame
                hands_crop, _ = hand_detector.detect_and_crop_hands(best_frame)

                if hands_crop is None:
                    print(f"No hands detected in best frame of {video_path}")
                    no_hands_detected += 1
                    continue

                # Convert to PIL Image and resize
                image = Image.fromarray(cv2.cvtColor(hands_crop, cv2.COLOR_BGR2RGB))
                image = image.resize((224, 224), Image.LANCZOS)

                # Save processed image in class subdirectory
                save_path = os.path.join(output_dir, split, gloss, f'{video_id}.jpg')
                image.save(save_path)

                total_processed += 1
                split_counts[split] += 1
                class_counts[split][gloss] += 1

            except Exception as e:
                print(f"Error processing {video_path}: {str(e)}")
                total_failed += 1
                continue

    # Print initial statistics
    print("\nInitial Processing completed!")
    print(f"Total videos processed successfully: {total_processed}")
    print(f"Total videos failed: {total_failed}")
    print(f"Videos with no hands detected: {no_hands_detected}")
    print("\nInitial split distribution:")
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

    # Step 1: Create dataset with hand detection
    create_image_dataset(
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        frames_to_sample=5  # Sample 5 frames from each video
    )

    # Step 2: Clean up and validate dataset
    stats = cleanup_and_validate_dataset()

    # Step 3: Suggest fixes for any issues
    suggest_fixes(stats)

    print("\nDataset processing complete!")
    print("The dataset now contains:")
    print("1. Frames with best hand detections")
    print("2. Consistent classes across all splits")
    print("3. No empty folders")


if __name__ == "__main__":
    main()