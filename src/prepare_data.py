# Task 2: Data Preparation for Emotion Detection
# Dataset: dair-ai/emotion (~20,000 tweets, 6 emotion labels)

from datasets import load_dataset
from collections import Counter
import json
import os

# ── Step 1: Download the dataset from Hugging Face ─────────────────
print("Downloading dataset from Hugging Face...")
ds = load_dataset('dair-ai/emotion')
print("Download complete!")
print(ds)

# ── Step 2: Inspect the raw data ───────────────────────────────────
print("\n--- First 3 rows of training data ---")
print(ds['train']['text'][:3])
print(ds['train']['label'][:3])

# ── Step 3: Check class distribution ───────────────────────────────
print("\n--- Class distribution (training set) ---")
counts = Counter(ds['train']['label'])
label_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
for label_id, count in sorted(counts.items()):
    print(f"  Label {label_id} ({label_names[label_id]}): {count} samples")

# ── Step 4: Clean the text (lowercase + strip whitespace) ──────────
print("\n--- Cleaning text data ---")
def clean_text(example):
    example['text'] = example['text'].lower().strip()
    return example

ds = ds.map(clean_text)
print("Cleaning done. Sample after cleaning:", ds['train'][0])

# ── Step 5: Create and save id2label.json ──────────────────────────
os.makedirs('data', exist_ok=True)
id2label = {
    "0": "sadness",
    "1": "joy",
    "2": "love",
    "3": "anger",
    "4": "fear",
    "5": "surprise"
}
with open('data/id2label.json', 'w') as f:
    json.dump(id2label, f, indent=2)
print("Saved: data/id2label.json")

# ── Step 6: Save train and test data as CSV files ──────────────────
import pandas as pd
train_df = pd.DataFrame(ds['train'])
test_df  = pd.DataFrame(ds['test'])
train_df.to_csv('data/train.csv', index=False)
test_df.to_csv('data/test.csv',  index=False)
print(f"Saved: data/train.csv ({len(train_df)} rows)")
print(f"Saved: data/test.csv  ({len(test_df)} rows)")

print("\nTask 2 complete! Ready to move to Task 3.")