import os
import json
import re
import shutil

# Hardcoded paths
video_dir = r"f:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\data\WLASL100"
json_path = r"f:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\data\WLASL100.json"

# Load JSON
def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

data = load_json(json_path)

yt_regex = re.compile(r"(?:v=|be/|embed/|youtu\\.be/|youtube\\.com/watch\\?v=)([A-Za-z0-9_-]{11})")
yt_id_to_videoid = {}

for entry in data:
    for inst in entry.get("instances", []):
        url = inst.get("url", "")
        video_id = str(inst["video_id"])
        # Try to extract YouTube ID from the URL
        match = yt_regex.search(url)
        if match:
            ytid = match.group(1)
            yt_id_to_videoid[ytid] = video_id

# Go through files in the directory
for fname in os.listdir(video_dir):
    base, ext = os.path.splitext(fname)
    if base in yt_id_to_videoid:
        new_name = yt_id_to_videoid[base] + ext
        src = os.path.join(video_dir, fname)
        dst = os.path.join(video_dir, new_name)
        if not os.path.exists(dst):
            print(f"Renaming {fname} -> {new_name}")
            shutil.move(src, dst)
        else:
            print(f"Target {new_name} already exists, skipping {fname}")

print("Done renaming YouTube ID files to video_id names.")
