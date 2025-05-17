import json
import cv2
from pathlib import Path

# —— CONFIGURE THESE ——
DATA_DIR    = Path(r'F:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\data\raw_videos_mp4\WLASL300')
JSON_ANN    = Path(r'F:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\data\WLASL_v0.3.json')
OUTPUT_DIR  = Path(r'F:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\data\processed')
TARGET_SIZE = (224, 224)
# ——————————————

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

        cap.set(cv2.CAP_PROP_POS_FRAMES, s_frame)
        x, y, w, h = info['bbox']

        # make output folder under the gloss name
        out_dir = OUTPUT_DIR / gloss / vid_id
        out_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for _ in range(s_frame, e_frame):
            ret, frame = cap.read()
            if not ret:
                break
            crop = frame[y:y+h, x:x+w]
            if crop.size == 0:
                print(f"    ⚠️ zero‐sized crop at frame {s_frame+saved}")
                continue
            img = cv2.resize(crop, TARGET_SIZE)
            cv2.imwrite(str(out_dir / f"{saved:06d}.jpg"), img)
            saved += 1

        cap.release()
        print(f"  • [{vid_id}] → {gloss} saved {saved} frames")

print("\n✅ All done. Frames are now organized by gloss.") 
