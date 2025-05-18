import json
import cv2
import numpy as np
from pathlib import Path

# —— CONFIGURE THESE ——
DATA_DIR    = Path(r'C:\video-asl-recognition\wordlevelrecogntion\data\raw_videos_mp4\WLASL300')
JSON_ANN    = Path(r'C:\video-asl-recognition\wordlevelrecogntion\data\WLASL_v0.3.json')
OUTPUT_DIR  = Path(r'C:\video-asl-recognition\wordlevelrecogntion\data\processed')
TARGET_SIZE = (224, 224)
NUM_SAMPLES = 16
# ————————————————

# 1) load & flatten JSON into dict: video_info[vid_id] → { gloss, start, end, bbox, fps }
with JSON_ANN.open('r', encoding='utf-8') as f:
    raw = json.load(f)

video_info = {}
for entry in raw:
    gloss = entry['gloss']
    for inst in entry['instances']:
        vid = inst['video_id']
        x1, y1, x2, y2 = inst['bbox']
        video_info[vid] = {
            'gloss': gloss,
            'start': inst['frame_start'],
            'end':   inst['frame_end'],
            'bbox':  (x1, y1, x2 - x1, y2 - y1),
            'fps':   inst.get('fps', None)
        }

# 2) Walk every numbered folder & video, use JSON gloss for output
for folder in sorted(DATA_DIR.iterdir(), key=lambda p: p.name):
    if not folder.is_dir():
        continue

    print(f"\n=== Scanning folder {folder.name} …")
    for vid_file in folder.glob('*.mp4'):
        vid_id = vid_file.stem
        info = video_info.get(vid_id)
        if info is None:
            print(f"  • [{vid_id}] not found in JSON → skipping")
            continue

        gloss = info['gloss']
        cap = cv2.VideoCapture(str(vid_file))
        if not cap.isOpened():
            print(f"  • failed to open {vid_file.name}")
            continue

        # fps fallback
        fps = info['fps'] or cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # determine frame range
        s = info['start']
        e = info['end'] if info['end'] >= 0 else total
        s_frame = max(0, min(int(s), total - 1))
        e_frame = max(s_frame + 1, min(int(e), total))

        if s_frame >= e_frame:
            print(f"  • [{vid_id}] bad range {s_frame}–{e_frame}")
            cap.release()
            continue

        # make output folder under the gloss name
        out_dir = OUTPUT_DIR / gloss / vid_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # grab exactly NUM_SAMPLES frames uniformly between s_frame and e_frame
        x, y, w, h = info['bbox']
        indices = np.linspace(s_frame, e_frame - 1, NUM_SAMPLES, dtype=int)

        saved = 0
        for target_frame in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(target_frame))
            ret, frame = cap.read()
            if not ret:
                print(f"    ⚠️ failed to read frame {target_frame}")
                continue
            crop = frame[y:y+h, x:x+w]
            if crop.size == 0:
                print(f"    ⚠️ zero-sized crop at frame {target_frame}")
                continue
            img = cv2.resize(crop, TARGET_SIZE)
            cv2.imwrite(str(out_dir / f"{saved:06d}.jpg"), img)
            saved += 1

        cap.release()
        print(f"  • [{vid_id}] → {gloss} saved {saved} frames")

print("\n✅ All done. Frames are now organized by gloss.")
