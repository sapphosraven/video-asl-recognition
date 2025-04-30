import os
import subprocess
from tqdm import tqdm

# Set your source and destination directories
SOURCE_DIR = 'videos'
DEST_DIR = 'videos'  # Set to SOURCE_DIR if you want to overwrite

# Supported input formats
VIDEO_EXTENSIONS = ['.mov', '.avi', '.webm', '.mkv', '.flv', '.mpg', '.mpeg', '.wmv', '.3gp', '.m4v']

def convert_video_to_mp4(input_path, output_path):
    # ffmpeg command to convert video to mp4 using H.264
    command = [
        'ffmpeg',
        '-i', input_path,
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-c:a', 'aac',
        '-strict', 'experimental',
        '-y',  # overwrite output
        output_path
    ]
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)

    video_files = []
    for root, _, files in os.walk(SOURCE_DIR):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in VIDEO_EXTENSIONS:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, SOURCE_DIR)
                video_files.append((full_path, rel_path))

    print(f"Found {len(video_files)} videos to convert.")

    for input_path, rel_path in tqdm(video_files, desc="Converting videos"):
        new_rel_path = os.path.splitext(rel_path)[0] + '.mp4'
        output_path = os.path.join(DEST_DIR, new_rel_path)
        output_dir = os.path.dirname(output_path)

        os.makedirs(output_dir, exist_ok=True)

        success = convert_video_to_mp4(input_path, output_path)
        if not success:
            print(f"\nFailed to convert: {input_path}")

    print("\nConversion complete!")

if __name__ == '__main__':
    main()
