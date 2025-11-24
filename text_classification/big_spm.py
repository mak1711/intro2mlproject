import random
import pandas as pd
import sentencepiece as spm

# === PARAMETERS ===
SPM_VOCAB_SIZE = 1500
SPM_MODEL_PREFIX = "spm_en_ar_joint"
TMP_TEXT_FILE = "spm_input.txt"
FINE_TUNE_INPUT_CSV = "/home/kan/ML/data/updated_data2.csv"  # your commands CSV
EN_FILE = "/home/kan/ML/data/eng_sentences.tsv"
AR_FILE = "/home/kan/ML/data/ara_sentences.tsv"

# === STEP 1: Load English + Arabic sentences from TSV (no header) ===
df_en = pd.read_csv(EN_FILE, sep="\t", header=None)
df_ar = pd.read_csv(AR_FILE, sep="\t", header=None)

all_sentences = []

# Sentences are in third column (index 2)
for line in df_en[2]:
    line = str(line).strip()
    if len(line) > 20:
        all_sentences.append(line)

for line in df_ar[2]:
    line = str(line).strip()
    if len(line) > 20:
        all_sentences.append(line)

random.shuffle(all_sentences)

# === STEP 2: Write sentences to temporary file for SPM training ===
with open(TMP_TEXT_FILE, "w", encoding="utf-8") as f:
    for sent in all_sentences:
        f.write(sent + "\n")

# === STEP 3: Train initial SentencePiece model ===
spm.SentencePieceTrainer.Train(
    input=TMP_TEXT_FILE,
    model_prefix=SPM_MODEL_PREFIX,
    vocab_size=SPM_VOCAB_SIZE,
    model_type="unigram",
    character_coverage=1.0,
    shuffle_input_sentence=True,
)

print("Initial SPM model trained:", SPM_MODEL_PREFIX + ".model")

# === STEP 4: Fine-tune SPM with your command dataset ===
df_cmds = pd.read_csv(FINE_TUNE_INPUT_CSV)
if "text" not in df_cmds.columns:
    raise ValueError("CSV must have a 'text' column.")

with open("fine_tune_input_combined.txt", "w", encoding="utf-8") as f:
    # first write English + Arabic sentences
    for sent in all_sentences:
        f.write(sent + "\n")
    # then add commands
    for line in df_cmds["text"]:
        f.write(str(line).strip() + "\n")

# Train fine-tuned SPM
spm.SentencePieceTrainer.Train(
    input="fine_tune_input_combined.txt",
    model_prefix=SPM_MODEL_PREFIX + "_finetuned",
    vocab_size=SPM_VOCAB_SIZE,
    model_type="unigram",
    character_coverage=1.0,
    shuffle_input_sentence=True,
)

print("Fine-tuned SPM model:", SPM_MODEL_PREFIX + "_finetuned.model")

# === STEP 5: Tokenize your commands with the fine-tuned SPM ===
sp = spm.SentencePieceProcessor(model_file=SPM_MODEL_PREFIX + "_finetuned.model")

with open("commands_tokenized.txt", "w", encoding="utf-8") as fout:
    for line in df_cmds["text"]:
        pieces = sp.encode(str(line).strip(), out_type=str)
        fout.write(" ".join(pieces) + "\n")

print("Tokenization done, saved to commands_tokenized.txt")

