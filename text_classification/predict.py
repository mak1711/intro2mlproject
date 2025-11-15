import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import sys

# When running this file directly (python text_classification/predict.py), ensure
# the project root (parent of this package) is on sys.path so absolute imports work.
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
	# look for label_map.txt in same folder
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
	cp = torch.load(model_path, map_location=device)
	return cp


def prepare_model_from_checkpoint(cp, max_len: int, device: torch.device):
	# cp may contain either a token-level 'vocab' (itos list) or a 'spm_model' filename
	label_map = cp.get('label_map')
	if label_map is None:
		raise RuntimeError('Checkpoint does not contain `label_map`')
	# invert label_map to id->label
	id2label = {int(v): k for k, v in label_map.items()}
	num_classes = len(id2label)

	# sentencepiece case
	spm_model_name = cp.get('spm_model')
	if spm_model_name is not None:
		# spm model should be next to the checkpoint
		# try loading from same folder
		from text_classification.spm import load_spm
		# model_path is not available here, but caller knows model file path; we'll try to find it in cwd or package dir
		# Prefer file in same folder as checkpoint: the caller provides model path when calling load_checkpoint
		# So prepare_model_from_checkpoint should be called by main after setting cwd appropriately. We'll try a few locations.
		candidates = [Path(spm_model_name), Path(__file__).resolve().parent.parent / spm_model_name]
		spm_path = None
		for c in candidates:
			if c.exists():
				spm_path = str(c)
				break
		if spm_path is None:
			# last resort: assume model stored in current working directory
			c = Path(spm_model_name)
			if c.exists():
				spm_path = str(c)
		if spm_path is None:
			raise FileNotFoundError(f"Could not find spm model file: {spm_model_name}")
		sp = load_spm(spm_path)
		# create thin vocab wrapper
		class _SPVocab:
			def __init__(self, sp):
				self.sp = sp
			def __len__(self):
				return int(self.sp.GetPieceSize())
			def encode(self, toks):
				if isinstance(toks, list):
					text = ' '.join(toks)
				else:
					text = toks
				return list(self.sp.EncodeAsIds(text))
			def stoi(self):
				return {}
		vocab = _SPVocab(sp)
	else:
		itos = cp.get('vocab')
		if itos is None:
			raise RuntimeError('Checkpoint does not contain `vocab` (itos list) and no spm_model info')
		vocab = build_vocab_from_itos(itos)

	model = TransformerClassifier(vocab_size=len(vocab), num_classes=num_classes, pad_idx=0, max_len=max_len)
	model.load_state_dict(cp['model_state'])
	model.to(device)
	model.eval()
	return model, vocab, id2label


def predict_text(model, vocab, id2label, text: str, device: torch.device, max_len: int = 32, topk: int = 3):
	toks = simple_tokenize(text)
	ids = vocab.encode(toks)[:max_len]
	if len(ids) < max_len:
		ids = ids + [vocab.stoi.get('<pad>', 0)] * (max_len - len(ids))
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
	p.add_argument('text', nargs='*', help='optional text to predict (if omitted, enters interactive mode)')
	args = p.parse_args()

	model_path = Path(args.model)
	device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	if not model_path.exists():
		raise FileNotFoundError(f'Checkpoint not found: {model_path}')

	cp = load_checkpoint(str(model_path), device)

	# some checkpoints may not include label_map in the state dict, try fallback
	if 'label_map' not in cp:
		lm = try_load_label_map_from_txt(model_path)
		if lm is not None:
			cp['label_map'] = lm

	model, vocab, id2label = prepare_model_from_checkpoint(cp, args.max_len, device)

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
