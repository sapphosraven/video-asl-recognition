import json
import csv

with open("sentence_reconstruction/data/input_target_pairs.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("sentence_reconstruction/data/t5_train.csv", "w", newline='', encoding="utf-8") as out_file:
    writer = csv.writer(out_file)
    writer.writerow(["source", "target"])
    for pair in data:
        input_text = "translate ASL to English: " + " ".join(pair["input"])
        output_text = pair["target"]
        writer.writerow([input_text, output_text])

print("✅ T5 training data saved to t5_train.csv")
