# Task 6: Inference script for emotion detection
# Reads input from environment variables so Docker can pass them in

import os
from transformers import pipeline
from huggingface_hub import login

# Read configuration from environment variables
HF_TOKEN    = os.getenv('HF_TOKEN', None)
HF_MODEL    = os.getenv('HF_MODEL_NAME',
              'g25ait2149/mlops-group28-emotion-distilbert-iitj')
INPUT_TEXT  = os.getenv('INPUT_TEXT', 'I am feeling happy today!')

# Login to Hugging Face if token provided
if HF_TOKEN:
    login(token=HF_TOKEN)

print(f"Loading model: {HF_MODEL}")
print(f"Input text   : {INPUT_TEXT}")
print("-" * 45)

# Load the classifier pipeline
classifier = pipeline(
    'text-classification',
    model=HF_MODEL
)

# Run inference
result = classifier(INPUT_TEXT)[0]

print(f"Predicted emotion : {result['label']}")
print(f"Confidence score  : {result['score']:.4f}")
print("-" * 45)
print("Inference complete!")