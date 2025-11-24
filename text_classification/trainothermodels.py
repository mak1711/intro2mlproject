import argparse
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
import sentencepiece as spm
import pandas as pd
from torch.utils.data import Dataset

from .othermodels import TextCNN, BiLSTM, CNN_BiLSTM  # relative import

device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------
# Dataset using SPM
# -------------------------------
class TextDatasetSPM(Dataset):
    def __init__(self, csv_path, spm_model_path, text_col="text", label_col="label", max_len=32):
        self.df = pd.read_csv(csv_path)
        self.texts = self.df[text_col].astype(str).tolist()
        self.labels = self.df[label_col].astype(int).tolist()
        self.max_len = max_len

        # Load sentencepiece model
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(spm_model_path)
        self.vocab_size = self.sp.get_piece_size()
        self.num_classes = len(set(self.labels))

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
# Training and evaluation functions
# -------------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(preds, 1)
        correct += (predicted == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = criterion(preds, y)

            total_loss += loss.item()
            _, predicted = torch.max(preds, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return total_loss / len(loader), correct / total


# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/updated_data2.csv")
    parser.add_argument("--spm-model", type=str, default="/home/kan/ML/spm_en_ar_joint.model")
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-filters", type=int, default=128)
    parser.add_argument("--kernel-sizes", type=str, default="3,4,5")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--save-dir", type=str, default="trained_models")
    args = parser.parse_args()

    kernel_sizes = list(map(int, args.kernel_sizes.split(",")))

    # Load dataset
    dataset = TextDatasetSPM(args.csv, args.spm_model, max_len=args.max_len)

    # -------------------------------
    # Split train / validation
    # -------------------------------
    indices = list(range(len(dataset)))
    labels = [dataset[i][1].item() for i in indices]  # convert tensor to int
    train_idx, val_idx = train_test_split(indices, test_size=0.1, stratify=labels, random_state=42)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False)

    vocab_size = dataset.vocab_size
    num_classes = dataset.num_classes

    # Timestamped folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(args.save_dir) / timestamp
    base_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------
    # Train helper
    # -------------------------------
    def run_training(model_class, model_name, **kwargs):
        model_dir = base_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n🔥 Training {model_name}...")
        model = model_class(**kwargs).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        train_acc_list, val_acc_list = [], []

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = evaluate(model, val_loader, criterion)

            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)

            print(f"Epoch {epoch}/{args.epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save model + accuracy
        torch.save({
            "model_state": model.state_dict(),
            "vocab_size": vocab_size,
            "num_classes": num_classes,
            "embed_dim": kwargs.get("embed_dim"),
            "num_filters": kwargs.get("num_filters"),
            "hidden_size": kwargs.get("hidden_size"),
            "kernel_sizes": kwargs.get("kernel_sizes"),
            "spm_model": args.spm_model,
            "train_acc": train_acc_list,
            "val_acc": val_acc_list
        }, model_dir / f"{model_name.lower()}.pth")
        print(f"{model_name} saved in {model_dir}")

    # -------------------------------
    # Run all models
    # -------------------------------
    run_training(TextCNN, "TextCNN",
                 vocab_size=vocab_size,
                 embed_dim=args.embed_dim,
                 num_filters=args.num_filters,
                 kernel_sizes=kernel_sizes,
                 num_classes=num_classes)

    run_training(BiLSTM, "BiLSTM",
                 vocab_size=vocab_size,
                 embed_dim=args.embed_dim,
                 hidden_size=args.hidden_size,
                 num_classes=num_classes)

    run_training(CNN_BiLSTM, "CNN_BiLSTM",
                 vocab_size=vocab_size,
                 embed_dim=args.embed_dim,
                 num_filters=args.num_filters // 2,
                 hidden_size=args.hidden_size,
                 kernel_sizes=kernel_sizes,
                 num_classes=num_classes)

    print(f"\nAll models saved under {base_dir}")


if __name__ == "__main__":
    main()

