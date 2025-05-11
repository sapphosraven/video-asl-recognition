from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, Seq2SeqTrainingArguments
from datasets import load_dataset, Dataset
import pandas as pd
import torch

# Load data
df = pd.read_csv("sentence_reconstruction/data/t5_train.csv")
dataset = Dataset.from_pandas(df)

# Tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)

def preprocess(example):
    inputs = tokenizer(example["source"], max_length=32, padding="max_length", truncation=True)
    targets = tokenizer(example["target"], max_length=32, padding="max_length", truncation=True)
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess)

# Model
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./t5_asl_model",
    per_device_train_batch_size=8,
    learning_rate=3e-4,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=10,
    predict_with_generate=True,
    save_strategy="epoch"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

trainer.train()

# Save model
model.save_pretrained("sentence_reconstruction/t5_asl_model")
tokenizer.save_pretrained("sentence_reconstruction/t5_asl_model")
print("✅ Fine-tuning complete and model saved.")
