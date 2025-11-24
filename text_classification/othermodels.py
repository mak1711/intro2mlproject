import argparse
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
import pandas as pd
import sentencepiece as spm
import torch.nn.functional as F

# -----------------------------
# Device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Dataset using SPM
# -----------------------------
class TextDatasetSPM(torch.utils.data.Dataset):
    def __init__(self, csv_path, spm_model_path, text_col="text", label_col="label", max_len=32):
        self.df = pd.read_csv(csv_path)
        self.texts = self.df[text_col].astype(str).tolist()
        self.labels = self.df[label_col].astype(int).tolist()
        self.max_len = max_len

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(spm_model_path)
        self.vocab_size = self.sp.get_piece_size()
        self.num_classes = len(set(self.labels))

    def encode_text(self, text):
        ids = self.sp.encode(text, out_type=int)
        # pad or truncate
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


# -----------------------------
# Models
# -----------------------------
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes,
                 num_filters=128, kernel_sizes=[3,4,5], dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        conv_outs = [F.relu(conv(x)) for conv in self.convs]
        pooled = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in conv_outs]
        x = torch.cat(pooled, dim=1)
        x = self.dropout(x)
        return self.fc(x)


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes, num_layers=1, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers,
                            bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)


class CNN_BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters,
                 hidden_size, num_classes, kernel_sizes=[3,4,5], dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k, padding=k//2) for k in kernel_sizes])
        self.lstm = nn.LSTM(num_filters * len(kernel_sizes), hidden_size,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        convs = [F.relu(conv(x)) for conv in self.convs]
        pooled = [F.adaptive_max_pool1d(c, x.size(2)) for c in convs]
        x = torch.cat(pooled, dim=1).transpose(1, 2)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)


# -----------------------------
# Training & evaluation
# -----------------------------
def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in train_loader:
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
    return total_loss / len(train_loader), correct / total


def evaluate(model, val_loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = criterion(preds, y)
            total_loss += loss.item()
            _, predicted = torch.max(preds, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return total_loss / len(val_loader), correct / total


# -----------------------------
# Main
# -----------------------------
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

    # Dataset + Train/Val split
    dataset = TextDatasetSPM(args.csv, args.spm_model, max_len=args.max_len)
    indices = list(range(len(dataset)))
    labels = [dataset[i][1].item() for i in indices]
    train_idx, val_idx = train_test_split(indices, test_size=0.1, stratify=labels, random_state=42)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False)

    vocab_size = dataset.vocab_size
    num_classes = dataset.num_classes

    # Timestamped folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(args.save_dir) / timestamp
    base_dir.mkdir(parents=True, exist_ok=True)

    # Helper to train & save a model
    def run_training(ModelClass, name, **kwargs):
        print(f"\n🔥 Training {name}...")
        model_dir = base_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)

        model = ModelClass(**kwargs).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        train_acc_list, val_acc_list = [], []
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = evaluate(model, val_loader, criterion)
            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)
            print(f"Epoch {epoch}/{args.epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save model & accuracies
        torch.save({
            "model_state": model.state_dict(),
            "vocab_size": vocab_size,
            "num_classes": num_classes,
            "train_acc": train_acc_list,
            "val_acc": val_acc_list,
            **kwargs
        }, model_dir / f"{name.lower()}.pth")
        print(f"{name} saved in {model_dir}")

    # Train all models
    run_training(TextCNN, "TextCNN",
                 vocab_size=vocab_size, embed_dim=args.embed_dim,
                 num_filters=args.num_filters, kernel_sizes=kernel_sizes,
                 num_classes=num_classes)

    run_training(BiLSTM, "BiLSTM",
                 vocab_size=vocab_size, embed_dim=args.embed_dim,
                 hidden_size=args.hidden_size, num_classes=num_classes)

    run_training(CNN_BiLSTM, "CNN_BiLSTM",
                 vocab_size=vocab_size, embed_dim=args.embed_dim,
                 num_filters=args.num_filters // 2, hidden_size=args.hidden_size,
                 kernel_sizes=kernel_sizes, num_classes=num_classes)


if __name__ == "__main__":
    main()

