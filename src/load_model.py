# Task 3: Load DistilBERT model and tokenizer from Hugging Face
# Model: distilbert-base-uncased (66MB, perfect for Kaggle free GPU)

import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Step 1: Load the id2label mapping we created in Task 2 ──────────
with open('data/id2label.json', 'r') as f:
    id2label = json.load(f)

label2id = {v: k for k, v in id2label.items()}

print("Labels loaded:")
print(id2label)

# ── Step 2: Load the tokenizer ───────────────────────────────────────
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
print("Tokenizer loaded!")

# ── Step 3: Test the tokenizer with a sample sentence ────────────────
sample = "I am feeling very happy today!"
tokens = tokenizer(sample, truncation=True, padding=True, return_tensors='pt')
print(f"\nSample sentence: {sample}")
print(f"Token IDs: {tokens['input_ids']}")
print(f"Number of tokens: {tokens['input_ids'].shape[1]}")

# ── Step 4: Load the model with 6 output labels ──────────────────────
print("\nLoading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=6,
    id2label=id2label,
    label2id=label2id
)
print("Model loaded!")
print(f"Number of output labels: {model.config.num_labels}")
print(f"Label mapping: {model.config.id2label}")

# ── Step 5: Quick sanity check — run a forward pass ──────────────────
import torch
print("\nRunning a test forward pass...")
with torch.no_grad():
    outputs = model(**tokens)
print(f"Output logits shape: {outputs.logits.shape}")
print("Shape should be: torch.Size([1, 6]) — 1 sample, 6 emotion labels")
print("\nTask 3 complete! Model and tokenizer verified.")