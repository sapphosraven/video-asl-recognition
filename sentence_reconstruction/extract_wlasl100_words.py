import json

# Load WLASL100 gloss list
with open("sentence_reconstruction/data/WLASL100.json", "r") as f:
    data = json.load(f)

# Extract gloss words and save to text file
glosses = sorted(set(entry['gloss'].upper() for entry in data))

with open("sentence_reconstruction/data/wlasl100_words.txt", "w") as f:
    for word in glosses:
        f.write(f"{word}\n")

print(f"Saved {len(glosses)} gloss words to sentence_reconstruction/data/wlasl100_words.txt")
