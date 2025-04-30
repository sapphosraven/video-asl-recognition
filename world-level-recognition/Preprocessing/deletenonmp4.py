import os

# Set the directory to clean
TARGET_DIR = 'videos'

# List of video extensions to delete (excluding .mp4)
VIDEO_EXTENSIONS_TO_DELETE = ['.mov', '.avi', '.webm', '.mkv', '.flv',
                              '.mpg', '.mpeg', '.wmv', '.3gp', '.m4v']

def delete_non_mp4_videos():
    deleted_count = 0
    for root, _, files in os.walk(TARGET_DIR):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in VIDEO_EXTENSIONS_TO_DELETE:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

    print(f"\nDone. Deleted {deleted_count} non-MP4 video(s).")

if __name__ == '__main__':
    delete_non_mp4_videos()
