# Text Commands Classification (from-scratch)

This folder contains a minimal PyTorch pipeline to train a Transformer encoder classifier from scratch on your voice command CSV files.

Files:
- `dataset.py` - Dataset and tokenizer/vocab builder
- `model.py` - Transformer encoder classifier
- `train.py` - Training script with checkpointing and validation
- `utils.py` - helper utilities (positional encoding, collate)
- `requirements.txt` - minimum Python packages

Quick start (from `/home/kan/ML`):

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r text_classification/requirements.txt
```

2. Train using the CSV in `data/`:

```bash
python text_classification/train.py \
  --csv data/voice_commands_dataset_en_with_no_meaning.csv \
  --text-col text --label-col label \
  --out-dir models/transformer_from_scratch \
  --epochs 10
```

Notes:
- The trainer builds a simple word-level vocabulary from the CSV and trains a Transformer encoder from scratch (no pretrained models).
- By default sequences are truncated/padded to `max_len=32` and vocab size is limited to 10000.
- If you prefer to use `voice_command_dataset_10000.csv` use `--text-col command --label-col intent`.

Predict from a saved checkpoint

After training, use `text_classification/predict.py` to load a checkpoint and predict labels from text input.

Single-shot prediction:

```bash
python text_classification/predict.py --model models/test_run/best.pth "turn on the lights"
```

Interactive mode:

```bash
python text_classification/predict.py --model models/test_run/best.pth
# then type text lines and press Enter; leave blank to exit
```

Options:

Training with a multilingual dataset (English + Arabic)
----------------------------------------------------

This code now includes a more robust tokenizer that normalizes Unicode and
removes Arabic diacritics so you can train on datasets containing both English
and Arabic text. To train with your new CSV `data/commands_dataset.csv` run:

```bash
python text_classification/train.py \
  --csv data/commands_dataset.csv \
  --text-col text --label-col label \
  --out-dir models/transformer_from_scratch \
  --epochs 10
```

Notes:
- The tokenizer preserves Arabic letters and will strip diacritics (vowel marks) so variations with/without diacritics are treated consistently.
- Ensure your CSV is UTF-8 encoded. The trainer/dataset already open files using UTF-8.
- If you have mixed columns or different column names, pass `--text-col` and `--label-col` accordingly.

