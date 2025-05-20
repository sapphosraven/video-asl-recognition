import spacy
import json

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def get_asl_structure(sentence):
    doc = nlp(sentence)
    subject = ''
    verb = ''
    dobj = ''

    for token in doc:
        if 'subj' in token.dep_:
            subject = token.text
        elif token.pos_ == 'VERB' and token.dep_ == 'ROOT':
            verb = token.lemma_  # Use base form
        elif 'obj' in token.dep_:
            dobj = token.text

    if subject and verb and dobj:
        return [dobj.upper(), subject.upper(), verb.upper()]
    elif subject and verb:
        return [subject.upper(), verb.upper()]
    else:
        return []

def process_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    results = []
    for sentence in lines:
        asl_input = get_asl_structure(sentence)
        if asl_input:
            results.append({
                "input": asl_input,
                "target": sentence
            })

    return results

def save_as_json(pairs, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, indent=2)

# ==== Main ====
input_file = "sentence_reconstruction/data/filtered_english_sentences.txt"  # Change to your filename
output_file = "sentence_reconstruction/data/input_target_pairs.json"

pairs = process_sentences(input_file)
save_as_json(pairs, output_file)

print(f"✅ Saved {len(pairs)} input-target pairs to '{output_file}'")
