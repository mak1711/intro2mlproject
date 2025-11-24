import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import sys
import sentencepiece as spm

# Ensure project root is on sys.path
proj_root = str(Path(__file__).resolve().parent.parent)
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from text_classification.model import TransformerClassifier
from text_classification.dataset import Vocab
from text_classification.utils import simple_tokenize

def build_vocab_from_itos(itos):
    v = Vocab(tokens=[])
    v.itos = itos
    v.stoi = {t: i for i, t in enumerate(itos)}
    return v

def try_load_label_map_from_txt(model_path: Path):
    p = model_path.parent / 'label_map.txt'
    if not p.exists():
        return None
    d = {}
    with open(p, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                k = parts[0]
                try:
                    v = int(parts[1])
                except Exception:
                    v = parts[1]
                d[k] = v
    return d

def load_checkpoint(model_path: str, device: torch.device):
    return torch.load(model_path, map_location=device)

def prepare_model_from_checkpoint(cp, max_len: int, device: torch.device,
                                  d_model=128, nhead=8, ff_dim=512, num_layers=4,
                                  dropout=0.1):
    # Load label map
    label_map = cp.get('label_map')
    if label_map is None:
        raise RuntimeError('Checkpoint does not contain `label_map`')
    id2label = {int(v): k for k, v in label_map.items()}
    num_classes = len(id2label)

    # Load vocab
    itos = cp.get('vocab')
    if itos is None:
        # fallback: use SentencePiece model
        sp_model_path = 'spm_en_ar_joint.model'  # adjust path if needed
        sp = spm.SentencePieceProcessor(model_file=sp_model_path)

        class SPVocab:
            def __init__(self, sp):
                self.sp = sp
                self.stoi = {str(i): i for i in range(sp.get_piece_size())}
                self.itos = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]

            def encode(self, tokens):
                # encode tokens with SentencePiece
                text = ' '.join(tokens)
                return self.sp.encode(text, out_type=int)

        vocab = SPVocab(sp)
    else:
        vocab = build_vocab_from_itos(itos)

    # Build model using checkpoint hyperparameters
    model = TransformerClassifier(
        vocab_size=len(vocab.itos),
        num_classes=num_classes,
        pad_idx=0,
        max_len=max_len,
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=ff_dim,
        num_layers=num_layers,
        dropout=dropout
    )

    model.load_state_dict(cp['model_state'])
    model.to(device)
    model.eval()
    return model, vocab, id2label

def predict_text(model, vocab, id2label, text: str, device: torch.device, max_len: int = 32, topk: int = 3):
    toks = simple_tokenize(text)
    ids = vocab.encode(toks)[:max_len]
    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))  # pad with 0
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=-1).cpu().squeeze(0)
    topk = min(topk, probs.numel())
    vals, inds = torch.topk(probs, topk)
    results = [(id2label[int(i)], float(v)) for v, i in zip(vals, inds)]
    return results

def main():
    p = argparse.ArgumentParser(description='CLI predictor for trained text classifier')
    p.add_argument('--model', '-m', default='models/test_run/best.pth', help='path to checkpoint .pth')
    p.add_argument('--max-len', type=int, default=32, help='maximum sequence length (must match training)')
    p.add_argument('--device', default=None, help='torch device (cpu or cuda). Default: auto-detect')
    p.add_argument('--topk', type=int, default=3, help='how many top predictions to show')
    # hyperparameters
    p.add_argument('--d-model', type=int, default=128)
    p.add_argument('--nhead', type=int, default=8)
    p.add_argument('--ff-dim', type=int, default=512)
    p.add_argument('--num-layers', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('text', nargs='*', help='optional text to predict (if omitted, enters interactive mode)')
    args = p.parse_args()

    model_path = Path(args.model)
    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not model_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {model_path}')

    cp = load_checkpoint(str(model_path), device)

    # fallback for label_map
    if 'label_map' not in cp:
        lm = try_load_label_map_from_txt(model_path)
        if lm is not None:
            cp['label_map'] = lm

    model, vocab, id2label = prepare_model_from_checkpoint(
        cp,
        max_len=args.max_len,
        device=device,
        d_model=args.d_model,
        nhead=args.nhead,
        ff_dim=args.ff_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    )

    def do_predict(s: str):
        results = predict_text(model, vocab, id2label, s, device, max_len=args.max_len, topk=args.topk)
        print(f"Input: {s}")
        for lbl, prob in results:
            print(f"  {lbl}\t{prob:.4f}")

    if args.text:
        txt = ' '.join(args.text)
        do_predict(txt)
    else:
        try:
            while True:
                s = input('Enter text (blank to exit): ').strip()
                if not s:
                    break
                do_predict(s)
        except (KeyboardInterrupt, EOFError):
            print('\nExiting')

if __name__ == '__main__':
    main()

