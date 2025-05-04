import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Load WLASL100
with open("sentence_reconstruction/data/WLASL100.json", "r") as f:
    data = json.load(f)

# Extract all gloss words
gloss_words = [item['gloss'] for item in data]

# Generate ASL-style "sentences" (sequences of glosses)
def create_asl_sentences(gloss_list, num_samples=500):
    asl_sentences = []
    english_sentences = []
    for _ in range(num_samples):
        # Randomly pick 3 to 5 words
        selected_words = random.sample(gloss_list, random.randint(3, 5))
        asl_sentence = " ".join(selected_words)
        english_sentence = " ".join(selected_words)  # placeholder (you'll fix this later)
        asl_sentences.append(asl_sentence)
        english_sentences.append(english_sentence)
    return asl_sentences, english_sentences

asl_sentences, english_sentences = create_asl_sentences(gloss_words)

# Create vocabularies
def build_vocab(sentences):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for sentence in sentences:
        for word in sentence.split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab

asl_vocab = build_vocab(asl_sentences)
eng_vocab = build_vocab(english_sentences)

# Tokenize sentences
def tokenize(sentences, vocab):
    return [[vocab.get(word, vocab["<UNK>"]) for word in sentence.split()] for sentence in sentences]

asl_tokenized = tokenize(asl_sentences, asl_vocab)
eng_tokenized = tokenize(english_sentences, eng_vocab)

# Pad sequences
max_asl_len = max(len(seq) for seq in asl_tokenized)
max_eng_len = max(len(seq) for seq in eng_tokenized)

asl_tokenized = [seq + [0] * (max_asl_len - len(seq)) for seq in asl_tokenized]
eng_tokenized = [seq + [0] * (max_eng_len - len(seq)) for seq in eng_tokenized]

# PyTorch Dataset
class ASLDataset(Dataset):
    def __init__(self, input_seqs, target_seqs):
        self.input_seqs = input_seqs
        self.target_seqs = target_seqs

    def __len__(self):
        return len(self.input_seqs)

    def __getitem__(self, idx):
        return torch.tensor(self.input_seqs[idx], dtype=torch.long), torch.tensor(self.target_seqs[idx], dtype=torch.long)

# DataLoader
dataset = ASLDataset(asl_tokenized, eng_tokenized)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Test a batch
for batch_idx, (inputs, targets) in enumerate(dataloader):
    print("Inputs:", inputs)
    print("Targets:", targets)
    break
