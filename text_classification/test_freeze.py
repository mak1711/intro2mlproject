import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from text_classification.pretrained_models import get_pretrained_model, PRETRAINED_MODELS

# ---------------- Dataset Wrapper ----------------
class PretrainedModelDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"text": self.texts[idx], "label": int(self.labels[idx])}


class TokenizerCollator:
    def __init__(self, tokenizer, max_length=128, return_tensors='pt'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_tensors = return_tensors

    def __call__(self, batch):
        texts = [item['text'] for item in batch]
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        enc = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_length,
            return_tensors=self.return_tensors
        )
        enc['labels'] = labels
        return enc


# -------------------------------------------------
# TESTING FUNCTION
# -------------------------------------------------
def test_models(model_paths, test_csv_path, text_col='text', label_col='label',
                batch_size=16, max_len=128):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    df = pd.read_csv(test_csv_path)
    texts = df[text_col].tolist()
    labels = df[label_col].tolist()
    test_dataset = PretrainedModelDataset(texts, labels)

    # Folder for saving all results
    save_root = Path("freeze_results")
    save_root.mkdir(exist_ok=True)

    # To store accuracy for plotting later
    acc_table = {}

    for model_path in model_paths:
        model_path = Path(model_path)

        # Extract freeze number and model type correctly
        # Example path: distilbert/freeze_3/distilbert/best.pth
        freeze_dir = model_path.parent                  # distilbert/
        freeze_block = freeze_dir.parent                # freeze_3/
        model_type_dir = freeze_block.parent            # distilbert/

        model_type = model_type_dir.name.lower()        # "distilbert"
        freeze_name = freeze_block.name                 # "freeze_3"

        # Extract the number from freeze_3
        freeze_match = re.search(r"freeze_(\d+)", freeze_name)
        if not freeze_match:
            print(f"Skipping folder without freeze_* format: {freeze_name}")
            continue
        freeze_num = int(freeze_match.group(1))

        model_name = f"{model_type}_freeze_{freeze_num}"

        print(f"\n{'='*50}")
        print(f"Testing model: {model_name}")
        print(f"{'='*50}")

        if model_type not in PRETRAINED_MODELS:
            print(f"Skipping unknown model type: {model_type}")
            continue

        # Prepare and load model
        model_wrapper = get_pretrained_model(
            model_type, num_classes=len(set(labels)), device=device
        )
        model = model_wrapper.model
        tokenizer = model_wrapper.tokenizer

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        collator = TokenizerCollator(tokenizer, max_length=max_len)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, collate_fn=collator)

        # Evaluation
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in test_loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)

                labels_batch = batch.pop("labels")
                outputs = model(**batch)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())

        # Metrics
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")
        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds)

        print(f"Accuracy: {acc:.4f}")
        print(f"Weighted F1: {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)

        # ----- SAVE EVERYTHING -----
        model_save_dir = save_root / model_name
        model_save_dir.mkdir(exist_ok=True)

        # Save metrics
        with open(model_save_dir / "metrics.json", "w") as f:
            json.dump({"accuracy": acc, "weighted_f1": f1}, f, indent=4)

        # Save classification report
        with open(model_save_dir / "classification_report.txt", "w") as f:
            f.write(report)

        # Save confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(model_save_dir / "confusion_matrix.png")
        plt.close()

        # Save accuracy for the global plot
        if model_type not in acc_table:
            acc_table[model_type] = []
        acc_table[model_type].append((freeze_num, acc))

    # -------------------------------------------------
    # PLOT ACCURACY VS FREEZE NUMBER
    # -------------------------------------------------
    plt.figure(figsize=(8, 6))

    for model_type, values in acc_table.items():
        values = sorted(values, key=lambda x: x[0])  # sort by freeze number
        xs = [v[0] for v in values]
        ys = [v[1] for v in values]
        plt.plot(xs, ys, marker="o", label=model_type)

    plt.title("Accuracy vs Freeze Number")
    plt.xlabel("Freeze Number")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_root / "accuracy_vs_freeze.png")
    plt.close()

    return acc_table


# ---------------- Run script ----------------
if __name__ == "__main__":

    ROOT = Path("/home/kan/ML/models/freeze_experiments")

    # Collect all best.pth files
    model_paths = list(ROOT.rglob("best.pth"))

    print("Found models to test:")
    for p in model_paths:
        print(p)

    test_csv_path = "/home/kan/ML/data/test_set.csv"

    results = test_models(model_paths, test_csv_path)
    print("\nSummary:", results)

