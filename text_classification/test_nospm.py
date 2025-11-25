import os
import json
from pathlib import Path
import pandas as pd
import torch

from text_classification.model import TransformerClassifier
from text_classification.dataset import Vocab
from text_classification.utils import simple_tokenize
import torch.nn.functional as F

# -----------------------------
# CONFIG / DEFAULT HYPERPARAMS
# -----------------------------
TEST_CSV = "/home/kan/ML/data/test_set.csv"
MODELS_ROOT = "/home/kan/ML/models/nospm_search"
MAX_LEN = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# default hyperparams from your training config.txt
DEFAULT_HYPERPARAMS = {
    "d_model": 256,
    "nhead": 8,
    "ff_dim": 512,
    "num_layers": 4,
    "dropout": 0.0,
}

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def load_checkpoint(model_path: str, device: torch.device):
    return torch.load(model_path, map_location=device)

def try_load_label_map_from_txt(model_path: Path):
    p = model_path.parent / "label_map.txt"
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

def build_vocab_from_itos(itos):
    v = Vocab(tokens=[])
    v.itos = itos
    v.stoi = {t: i for i, t in enumerate(itos)}
    return v

def prepare_model_from_checkpoint(cp, max_len: int, device: torch.device,
                                  d_model=256, nhead=8, ff_dim=512, num_layers=4, dropout=0.0):
    # Load label map
    label_map = cp.get("label_map")
    if label_map is None:
        raise RuntimeError("Checkpoint does not contain `label_map`")
    id2label = {int(v): k for k, v in label_map.items()}
    num_classes = len(id2label)

    # Load vocab
    itos = cp.get("vocab")
    if itos is None:
        raise RuntimeError("Checkpoint does not contain `vocab`")
    vocab = build_vocab_from_itos(itos)

    # Build model
    model = TransformerClassifier(
        vocab_size=len(vocab),
        num_classes=num_classes,
        pad_idx=0,
        max_len=max_len,
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=ff_dim,
        num_layers=num_layers,
        dropout=dropout
    )

    model.load_state_dict(cp["model_state"])
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

def load_metrics_json(run_dir):
    metrics_path = os.path.join(run_dir, "metrics.json")
    if not os.path.isfile(metrics_path):
        return None
    with open(metrics_path, "r") as f:
        return json.load(f)

# -----------------------------
# EVALUATION
# -----------------------------
def evaluate_model(model_path, hyperparams, df_test):
    print(f"  Loading model: {model_path}")
    cp = load_checkpoint(str(model_path), DEVICE)

    # ensure label_map exists
    if "label_map" not in cp:
        lm = try_load_label_map_from_txt(Path(model_path))
        if lm is not None:
            cp["label_map"] = lm

    model, vocab, id2label = prepare_model_from_checkpoint(
        cp,
        max_len=MAX_LEN,
        device=DEVICE,
        d_model=hyperparams.get("d_model", DEFAULT_HYPERPARAMS["d_model"]),
        nhead=hyperparams.get("nhead", DEFAULT_HYPERPARAMS["nhead"]),
        ff_dim=hyperparams.get("ff_dim", DEFAULT_HYPERPARAMS["ff_dim"]),
        num_layers=hyperparams.get("num_layers", DEFAULT_HYPERPARAMS["num_layers"]),
        dropout=hyperparams.get("dropout", DEFAULT_HYPERPARAMS["dropout"]),
    )

    correct, total = 0, 0
    for _, row in df_test.iterrows():
        text = row["text"]
        true_label = int(row["label"])
        pred = predict_text(model, vocab, id2label, text, DEVICE, max_len=MAX_LEN, topk=1)
        predicted_label_name = pred[0][0]

        # convert label name → numeric ID
        predicted_label = None
        for label_id, label_name in id2label.items():
            if label_name == predicted_label_name:
                predicted_label = int(label_id)

        if predicted_label == true_label:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0

# -----------------------------
# MAIN
# -----------------------------
def main():
    df_test = pd.read_csv(TEST_CSV)
    model_runs = sorted(os.listdir(MODELS_ROOT))
    results = []

    for run in model_runs:
        run_dir = os.path.join(MODELS_ROOT, run)
        best_pth = os.path.join(run_dir, "best.pth")

        if not os.path.isfile(best_pth):
            print(f"Skipping {run}: best.pth not found")
            continue

        metrics = load_metrics_json(run_dir)
        hyperparams = metrics.get("hyperparams", {}) if metrics is not None else {}

        print(f"\nEvaluating {run} ...")
        try:
            acc = evaluate_model(best_pth, hyperparams, df_test)
            print(f"  Acc = {acc:.4f}")
            results.append((run, acc))
        except Exception as e:
            print(f"  Skipping {run} due to error: {e}")

    # Save results
    out_csv = "model_search2_results_nospm.csv"
    pd.DataFrame(results, columns=["run", "accuracy"]).to_csv(out_csv, index=False)
    print(f"\nSaved results → {out_csv}")

if __name__ == "__main__":
    main()

