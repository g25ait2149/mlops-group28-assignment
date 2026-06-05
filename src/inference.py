import os
from transformers import pipeline

HF_MODEL = os.getenv('HF_MODEL_NAME',
                     'g25ait2149/mlops-group28-emotion-distilbert-iitj')
INPUT_TEXT = os.getenv('INPUT_TEXT', 'I am feeling happy today!')

print(f"Loading model: {HF_MODEL}")
print(f"Input text   : {INPUT_TEXT}")
print("-" * 45)

classifier = pipeline(
    'text-classification',
    model=HF_MODEL
)

result = classifier(INPUT_TEXT)[0]

print(f"Predicted emotion : {result['label']}")
print(f"Confidence score  : {result['score']:.4f}")
print("-" * 45)
print("Inference complete!")