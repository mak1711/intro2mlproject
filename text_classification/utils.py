import math
import torch
import re
import unicodedata
from torch.nn import Module

class PositionalEncoding(Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return x


# token regex: keep word chars and Arabic block (basic range). This lets
# tokens include English and Arabic words. We also allow the apostrophe.
_token_re = re.compile(r"[^\w\u0600-\u06FF']+", flags=re.U)


def normalize_text(text: str) -> str:
    """Normalize unicode text: NFC/KC, remove tatweel and combining marks (diacritics), lower-case.

    This helps when handling Arabic text with optional diacritics and
    mixed English/Arabic input.
    """
    if not isinstance(text, str):
        text = str(text)
    # normalize compatibility characters
    text = unicodedata.normalize('NFKC', text)
    # remove tatweel (kashida) used to stretch Arabic words
    text = text.replace('\u0640', '')
    # remove combining marks (diacritics) - category 'Mn'
    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')
    return text.lower()


def simple_tokenize(text: str):
    text = normalize_text(text)
    tokens = [t for t in _token_re.split(text) if t]
    return tokens


def collate_batch(batch, pad_idx=0, max_len=32):
    # batch: list of (token_ids, label)
    xs = [b[0][:max_len] for b in batch]
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    batch_size = len(xs)
    seq_lens = [len(x) for x in xs]
    max_l = min(max(seq_lens), max_len)
    padded = torch.full((batch_size, max_l), pad_idx, dtype=torch.long)
    for i, x in enumerate(xs):
        l = min(len(x), max_l)
        padded[i, :l] = torch.tensor(x[:l], dtype=torch.long)
    return padded, ys
