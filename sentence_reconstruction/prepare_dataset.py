import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from collections import Counter
import numpy as np

# Load dataset
with open("sentence_reconstruction/data/WLASL100.json", "r") as f:
    data = json.load(f)

# Prepare input and target sentences
asl_sentences = [item['asl'] for item in data]  # ASL-style sentences
english_sentences = [item['english'] for item in data]  # English sentences

# Create vocabularies
def create_vocab(sentences):
    word_count = Counter()
    for sentence in sentences:
        word_count.update(sentence.split())
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(word_count.items())}
    vocab["<PAD>"] = 0  # Add padding token
    vocab["<UNK>"] = len(vocab)  # Add unknown token
    return vocab

asl_vocab = create_vocab(asl_sentences)
eng_vocab = create_vocab(english_sentences)

# Convert sentences to token ids
def sentence_to_token_ids(sentence, vocab):
    return [vocab.get(word, vocab["<UNK>"]) for word in sentence.split()]

asl_tokenized = [sentence_to_token_ids(sentence, asl_vocab) for sentence in asl_sentences]
eng_tokenized = [sentence_to_token_ids(sentence, eng_vocab) for sentence in english_sentences]

# Pad sequences for uniform length
def pad_sequences(sequences, max_len):
    return [seq + [0] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len] for seq in sequences]

max_asl_len = max(len(seq) for seq in asl_tokenized)
max_eng_len = max(len(seq) for seq in eng_tokenized)

asl_tokenized_padded = pad_sequences(asl_tokenized, max_asl_len)
eng_tokenized_padded = pad_sequences(eng_tokenized, max_eng_len)

# Create PyTorch dataset and DataLoader
class ASLTranslationDataset(Dataset):
    def __init__(self, input_data, target_data):
        self.input_data = input_data
        self.target_data = target_data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input_seq = torch.tensor(self.input_data[idx], dtype=torch.long)
        target_seq = torch.tensor(self.target_data[idx], dtype=torch.long)
        return input_seq, target_seq

# Create DataLoader for batching
dataset = ASLTranslationDataset(asl_tokenized_padded, eng_tokenized_padded)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=lambda batch: (
    pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0),
    pad_sequence([item[1] for item in batch], batch_first=True, padding_value=0)
))

# Sample data from the dataloader
for batch_idx, (input_batch, target_batch) in enumerate(dataloader):
    print(f"Batch {batch_idx+1}")
    print("Input:", input_batch)
    print("Target:", target_batch)
    break
