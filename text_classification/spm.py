import os
from pathlib import Path
import sentencepiece as spm


def train_sentencepiece_from_textfile(input_txt: str, model_prefix: str, vocab_size: int = 16000, model_type: str = 'unigram') -> str:
    """Train a SentencePiece model from a plain text file (one sentence per line).
    Returns the path to the trained .model file.
    """
    args = f"--input={input_txt} --model_prefix={model_prefix} --vocab_size={vocab_size} --model_type={model_type} --character_coverage=1.0"
    spm.SentencePieceTrainer.Train(args)
    model_path = f"{model_prefix}.model"
    if not Path(model_path).exists():
        raise FileNotFoundError(f"SentencePiece model not created: {model_path}")
    return model_path


def load_spm(model_path: str):
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    return sp


def save_spm_to_dir(model_path: str, out_dir: str):
    """Copy the .model and .vocab files to out_dir, return new model filename (relative to out_dir)"""
    p = Path(model_path)
    dest_dir = Path(out_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for ext in ['.model', '.vocab']:
        src = p.with_suffix(ext)
        if src.exists():
            dst = dest_dir / src.name
            with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
                fdst.write(fsrc.read())
    return p.name
