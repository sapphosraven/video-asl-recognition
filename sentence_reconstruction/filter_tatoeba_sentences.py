import csv

# Load WLASL100 words
with open("sentence_reconstruction/data/wlasl100_words.txt", "r") as f:
    wlasl_words = set(word.strip().lower() for word in f)

# Stopwords to ignore
stopwords = {
    "a", "an", "the", "is", "are", "was", "were", "to", "of", "in", "on", "at",
    "with", "and", "or", "for", "by", "this", "that", "those", "as", "it", "be",
    "but", "if", "not", "do", "does", "did", "have", "has", "had", "will", "would", "can"
}

filtered_sentences = []
rejected_count = 0

with open("sentence_reconstruction/data/sentences.tsv", "r", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile, delimiter="\t")
    for row in reader:
        if len(row) != 3:
            continue

        sentence_id, lang, sentence = row
        if lang != "eng":
            continue

        # Clean and tokenize
        words = sentence.strip().lower().replace(".", "").replace(",", "").split()
        content_words = [word for word in words if word not in stopwords]

        if not content_words:
            continue

        match_count = sum(word in wlasl_words for word in content_words)
        match_ratio = match_count / len(content_words)

        if match_ratio >= 0.5:
            filtered_sentences.append(sentence.strip())
        else:
            rejected_count += 1

print(f"✔️ Found {len(filtered_sentences)} valid English sentences.")
print(f"❌ Rejected {rejected_count} sentences.")

# Save only the English sentences
with open("sentence_reconstruction/data/filtered_english_sentences.txt", "w", encoding="utf-8") as f:
    for s in filtered_sentences:
        f.write(s + "\n")

print("✅ Saved to sentence_reconstruction/data/filtered_english_sentences.txt")
