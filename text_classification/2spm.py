import pandas as pd
import sentencepiece as spm
from pathlib import Path

# Paths
csv_path = "data/augmented.csv"
output_dir = Path("spm_model2")
output_dir.mkdir(exist_ok=True)

# Step 1: Load CSV
print("Loading CSV...")
df = pd.read_csv(csv_path)

if "text" not in df.columns:
    raise ValueError("CSV must contain a column named 'text'.")

# Step 2: Save text column to a plain text file
text_file = output_dir / "all_text.txt"
print("Extracting text column...")
df["text"].to_csv(text_file, index=False, header=False)

# Step 3: Train SentencePiece model
model_prefix = output_dir / "spm_unigram_8000"

print("Training SentencePiece model...")
spm.SentencePieceTrainer.Train(
    input=str(text_file),
    model_prefix=str(model_prefix),
    vocab_size=1000,
    model_type="unigram",         # better for multilingual text
    character_coverage=1.0,       # full character coverage (Arabic + English)
    input_sentence_size=1000000,  # optional: limit if dataset huge
    shuffle_input_sentence=True
)

print("\nDone!")
print(f"Generated files:")
print(f"- {model_prefix}.model")
print(f"- {model_prefix}.vocab")

