import csv
import json
from collections import Counter
from typing import List, Optional
from pathlib import Path
from .utils import simple_tokenize
from pathlib import Path as PPath

try:
    from text_classification.spm import load_spm
except Exception:
    load_spm = None

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'

class Vocab:
    def __init__(self, tokens=None, max_size=10000, min_freq=1):
        self.counter = Counter(tokens or [])
        self.max_size = max_size
        self.min_freq = min_freq
        self.stoi = {}
        self.itos = []
        self.build()

    def build(self):
        freq = [(tok, c) for tok, c in self.counter.items() if c >= self.min_freq]
        freq.sort(key=lambda x: (-x[1], x[0]))
        toks = [t for t, _ in freq][: self.max_size]
        self.itos = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN] + toks
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens: List[str]):
        return [self.stoi.get(t, self.stoi[UNK_TOKEN]) for t in tokens]

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'itos': self.itos}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        obj = cls(tokens=[])
        obj.itos = data['itos']
        obj.stoi = {t: i for i, t in enumerate(obj.itos)}
        return obj


class TextCommandsDataset:
    def __init__(self, csv_path, text_col='text', label_col='label', vocab: Optional[Vocab]=None, build_vocab=False, max_vocab=10000, spm_model: Optional[str]=None):
        self.csv_path = csv_path
        self.text_col = text_col
        self.label_col = label_col
        self.rows = []
        self._labels = []
        self._texts = []
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                if text_col not in r or label_col not in r:
                    # try alternates
                    if 'command' in r and 'intent' in r:
                        txt = r['command']
                        lbl = r['intent']
                    else:
                        txt = r.get(text_col, '')
                        lbl = r.get(label_col, 'no_meaning')
                else:
                    txt = r[text_col]
                    lbl = r[label_col]
                self._texts.append(txt)
                self._labels.append(lbl)
        # build label mapping
        self.labels = sorted(list(set(self._labels)))
        self.label2id = {l:i for i,l in enumerate(self.labels)}
        # if a SentencePiece model is provided, use it for encoding
        self.spm = None
        if spm_model is not None:
            if load_spm is None:
                raise RuntimeError('SentencePiece helpers not available. Ensure sentencepiece is installed and text_classification.spm exists')
            # support relative paths
            spm_path = str(spm_model)
            # if path is relative to data folder, try resolving
            if not PPath(spm_path).exists():
                # try same folder as csv
                spm_path_candidate = PPath(csv_path).resolve().parent / spm_path
                if spm_path_candidate.exists():
                    spm_path = str(spm_path_candidate)
            self.spm = load_spm(spm_path)
            # create a thin vocab-like wrapper around SPM so code that expects ds.vocab works
            class _SPVocab:
                def __init__(self, sp):
                    self.sp = sp
                def __len__(self):
                    return int(self.sp.GetPieceSize())
                def encode(self, toks):
                    # sentencepiece expects a string, but our pipeline may pass tokens list
                    if isinstance(toks, list):
                        text = ' '.join(toks)
                    else:
                        text = toks
                    return list(self.sp.EncodeAsIds(text))
                def save(self, path: str):
                    # nothing to do here; SPM model saved separately
                    pass
            self.vocab = _SPVocab(self.spm)
        else:
            # build or load vocab
            if vocab is None and build_vocab:
                tokens = []
                for t in self._texts:
                    tokens.extend(simple_tokenize(t))
                self.vocab = Vocab(tokens=tokens, max_size=max_vocab)
            elif vocab is None:
                # create minimal vocab from tokens seen
                tokens = []
                for t in self._texts:
                    tokens.extend(simple_tokenize(t))
                self.vocab = Vocab(tokens=tokens, max_size=max_vocab)
            else:
                self.vocab = vocab

    def __len__(self):
        return len(self._texts)

    def __getitem__(self, idx):
        txt = self._texts[idx]
        lbl = self._labels[idx]
        if self.spm is not None:
            # encode with sentencepiece
            ids = self.vocab.encode(txt)
        else:
            toks = simple_tokenize(txt)
            ids = self.vocab.encode(toks)
        return ids, self.label2id[lbl]

    def save_vocab(self, path: str):
        self.vocab.save(path)

    def get_label_map(self):
        return self.label2id

    # Utilities used by training script for augmentation
    def decode_text(self, id_list):
        """Convert a sequence of ids back to a string for augmentation/inspection."""
        if self.spm is not None:
            # sentencepiece DecodeIds expects a list of ints
            try:
                return self.spm.DecodeIds([int(x) for x in id_list if int(x) >= 0])
            except Exception:
                # fallback: join pieces
                try:
                    return ' '.join(self.spm.IdToPiece(int(x)) for x in id_list if int(x) >= 0)
                except Exception:
                    return ''
        else:
            pieces = []
            for i in id_list:
                try:
                    ii = int(i)
                except Exception:
                    continue
                if ii < 0 or ii >= len(self.vocab.itos):
                    continue
                tok = self.vocab.itos[ii]
                if tok in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN):
                    continue
                pieces.append(tok)
            return ' '.join(pieces)

    def encode_text(self, text: str, max_len: int = 32):
        """Encode a raw text string into padded/truncated id list of length <= max_len."""
        if self.spm is not None:
            ids = list(self.vocab.encode(text))
        else:
            toks = simple_tokenize(text)
            ids = self.vocab.encode(toks)
        ids = ids[:max_len]
        pad_idx = 0
        # try to get pad index from vocab if available
        try:
            pad_idx = self.vocab.stoi.get(PAD_TOKEN, 0)
        except Exception:
            pad_idx = 0
        if len(ids) < max_len:
            ids = ids + [pad_idx] * (max_len - len(ids))
        return ids
