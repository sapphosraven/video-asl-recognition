import json
import os

# Load the JSON data
with open('WLASL_v0.3.json') as f:
    data = json.load(f)

# Get all referenced video IDs from JSON
json_ids = {inst['video_id'] for entry in data for inst in entry['instances']}

# Get all actual video files (remove extensions)
video_dir = 'videos'
video_ids = {os.path.splitext(f)[0] for f in os.listdir(video_dir) if f.endswith('.mp4')}

# Compare
missing_in_json = video_ids - json_ids
missing_on_disk = json_ids - video_ids

# Print the results
print(f"Videos in folder but not in JSON: {len(missing_in_json)}")
if missing_in_json:
    print("\nMissing in JSON (but present in folder):")
    for video in missing_in_json:
        print(video)

print(f"\nVideos in JSON but missing from folder: {len(missing_on_disk)}")

