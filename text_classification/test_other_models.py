import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from pathlib import Path
from .othermodels import TextCNN, BiLSTM, CNN_BiLSTM  # your model definitions
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Dataset for test set
# -------------------------------
class TextDatasetSPM(Dataset):
    def __init__(self, csv_path, spm_model_path, text_col="text", label_col="label", max_len=32):
        self.df = pd.read_csv(csv_path)
        self.texts = self.df[text_col].astype(str).tolist()
        self.labels = self.df[label_col].astype(int).tolist()
        self.max_len = max_len

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(spm_model_path)

    def encode_text(self, text):
        ids = self.sp.encode(text, out_type=int)
        if len(ids) < self.max_len:
            ids += [0] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.encode_text(self.texts[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# -------------------------------
# Evaluation function
# -------------------------------
def evaluate_model(model, loader):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            _, predicted = torch.max(preds, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    acc = correct / total
    return acc, all_preds, all_labels


# -------------------------------
# Main testing
# -------------------------------
def main():
    test_csv = "/home/kan/ML/data/test_set.csv"
    spm_model = "/home/kan/ML/spm_en_ar_joint.model"
    batch_size = 64
    max_len = 32

    test_dataset = TextDatasetSPM(test_csv, spm_model, max_len=max_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Paths to your saved models (change to your actual folder)
    model_folder = Path("othermodels2/20251123_075844")  # replace with actual timestamp folder

    models_to_test = {
        "TextCNN": TextCNN,
        "BiLSTM": BiLSTM,
        "CNN_BiLSTM": CNN_BiLSTM
    }

    for model_name, ModelClass in models_to_test.items():
        model_path = model_folder / model_name / f"{model_name.lower()}.pth"
        checkpoint = torch.load(model_path, map_location=device)

        # Load model
        kwargs = {
            "vocab_size": checkpoint["vocab_size"],
            "embed_dim": checkpoint.get("embed_dim", 128),
            "num_classes": checkpoint["num_classes"],
        }

        if model_name == "TextCNN":
            kwargs.update({
                "num_filters": checkpoint.get("num_filters", 128),
                "kernel_sizes": checkpoint.get("kernel_sizes", [3,4,5])
            })
        elif model_name == "BiLSTM":
            kwargs.update({
                "hidden_size": checkpoint.get("hidden_size", 128)
            })
        elif model_name == "CNN_BiLSTM":
            kwargs.update({
                "num_filters": checkpoint.get("num_filters", 64),
                "hidden_size": checkpoint.get("hidden_size", 128),
                "kernel_sizes": checkpoint.get("kernel_sizes", [3,4,5])
            })

        model = ModelClass(**kwargs).to(device)
        model.load_state_dict(checkpoint["model_state"])

        acc, preds, labels = evaluate_model(model, test_loader)
        print(f"{model_name} Test Accuracy: {acc:.4f}")

        # -------------------------------
        # High-quality confusion matrix
        # -------------------------------
        cm = confusion_matrix(labels, preds)
        save_dir = model_folder / model_name / "plots"
        save_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{model_name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()

        # Save in high-quality formats
        plt.savefig(save_dir / f"{model_name}_confusion_matrix.png", dpi=400, bbox_inches="tight")
        plt.savefig(save_dir / f"{model_name}_confusion_matrix.pdf", bbox_inches="tight")
        plt.savefig(save_dir / f"{model_name}_confusion_matrix.svg", bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()

