import os
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from text_classification.model import TransformerClassifier
from text_classification.dataset import Vocab
from text_classification.utils import simple_tokenize

# -----------------------------
# CONFIG / MODEL
# -----------------------------
MODEL_PTH = "/home/kan/ML/models/nospm_search/mild_reg_3ep/best.pth"
CONFIG_TXT = "/home/kan/ML/models/nospm_search/mild_reg_3ep/config.txt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = Path("/home/kan/ML/plots")
SAVE_DIR.mkdir(exist_ok=True)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def read_config_txt(path):
    params = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                try:
                    if "." in val:
                        val = float(val)
                    else:
                        val = int(val)
                except ValueError:
                    pass
                params[key] = val
    return params

def load_checkpoint(model_path, device):
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
    label_map = cp.get("label_map")
    if label_map is None:
        raise RuntimeError("Checkpoint does not contain `label_map`")
    id2label = {int(v): k for k, v in label_map.items()}
    num_classes = len(id2label)

    itos = cp.get("vocab")
    if itos is None:
        raise RuntimeError("Checkpoint does not contain `vocab`")
    vocab = build_vocab_from_itos(itos)

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

# -----------------------------
# MAIN
# -----------------------------
def main():
    cfg = read_config_txt(CONFIG_TXT)
    test_csv = "data/test_set.csv"
    max_len = cfg.get("max_len", 32)

    #if not os.path.isfile(test_csv):
    #    raise FileNotFoundError(f"CSV not found: {test_csv}")
    df_test = pd.read_csv(test_csv)

    cp = load_checkpoint(MODEL_PTH, DEVICE)

    if "label_map" not in cp:
        lm = try_load_label_map_from_txt(Path(MODEL_PTH))
        if lm is not None:
            cp["label_map"] = lm

    model, vocab, id2label = prepare_model_from_checkpoint(
        cp,
        max_len=max_len,
        device=DEVICE,
        d_model=cfg.get("d_model", 256),
        nhead=cfg.get("nhead", 8),
        ff_dim=cfg.get("ff_dim", 512),
        num_layers=cfg.get("num_layers", 4),
        dropout=cfg.get("dropout", 0.0),
    )

    correct, total = 0, 0
    y_true, y_pred = [], []

    for _, row in df_test.iterrows():
        text = row["text"]
        true_label = int(row["label"])
        pred = predict_text(model, vocab, id2label, text, DEVICE, max_len=max_len, topk=1)
        predicted_label_name = pred[0][0]

        predicted_label = None
        for label_id, label_name in id2label.items():
            if label_name == predicted_label_name:
                predicted_label = int(label_id)

        y_true.append(true_label)
        y_pred.append(predicted_label)

        if predicted_label == true_label:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0.0
    print(f"\nModel Accuracy on {test_csv} = {acc:.4f}")

    # -----------------------------
    # CONFUSION MATRIX (HIGH QUALITY SAVE)
    # -----------------------------
    cm = confusion_matrix(y_true, y_pred)
    labels = [id2label[i] for i in sorted(id2label.keys())]

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    # Save in high quality
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "confusion_matrix.png", dpi=400, bbox_inches="tight")
    plt.savefig(SAVE_DIR / "confusion_matrix.pdf", bbox_inches="tight")
    plt.savefig(SAVE_DIR / "confusion_matrix.svg", bbox_inches="tight")
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()

