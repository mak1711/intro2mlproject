import argparse
import random
import os
from pathlib import Path
import torch
import torch.cuda.amp as amp
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import itertools
import json

from text_classification.dataset import TextCommandsDataset
from text_classification.model import TransformerClassifier
from text_classification.utils import collate_batch

# --------------------- Seed ---------------------
def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

# --------------------- Accuracy ---------------------
def accuracy(preds, y):
    return (preds.argmax(1) == y).float().mean().item()

# --------------------- Data augmentation ---------------------
def augment_text(text, lang='en', intensity=1.0):
    text = text.strip()
    if isinstance(lang, (int, float)):
        intensity = float(lang)
        lang = 'en'
    base_case_p = 0.2
    base_punct_p = 0.2
    base_syn_p = 0.5

    if random.random() < base_case_p * intensity:
        text = text.upper() if random.random() < 0.5 else text.lower()
    if random.random() < base_punct_p * intensity:
        text = text.replace('.', '').replace(',', '') + random.choice(['', '.', '!'])

    synonyms_en = {
        "go left": ["move to the left", "turn left", "left please"],
        "go right": ["move to the right", "turn right", "right please"],
        "go forward": ["move forward", "forward please", "advance"],
        "go back": ["move backward", "backwards please", "reverse"],
        "sit": ["take a seat", "sit down", "have a seat"],
        "stand": ["stand up", "rise", "get up"],
        "turn around": ["spin", "rotate", "turn 180"],
        "do a trick": ["perform a trick", "show trick", "do something cool"],
        "no meaning": ["nothing", "ignore this", "no command"]
    }
    synonyms_ar = {
        "اجلس": ["تفضل بالجلوس", "اقعد", "جلوس"],
        "قف": ["قف واقف", "انهض", "وقف"],
        "تقدم": ["تحرك للأمام", "تقدم للأمام"],
        "تراجع": ["ارجع", "تحرك للخلف"],
        "انعطف يمين": ["اتجه يمينا", "اتجه إلى اليمين"],
        "انعطف يسار": ["اتجه يسارا", "اتجه إلى اليسار"],
        "استدر": ["استدر حول نفسك", "ادور"],
        "قم بحركة": ["اعمل حركة", "قم بخدعة"],
        "بلا معنى": ["لا شيء", "تجاهل", "لا أمر"]
    }

    key = text.lower()
    if lang == 'en' and key in synonyms_en and random.random() < base_syn_p * intensity:
        text = random.choice(synonyms_en[key])
    if lang == 'ar' and key in synonyms_ar and random.random() < base_syn_p * intensity:
        text = random.choice(synonyms_ar[key])
    return text

# --------------------- Main ---------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--text-col', default='text')
    parser.add_argument('--label-col', default='label')
    parser.add_argument('--out-dir', default='models/transformer_aug')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-len', type=int, default=32)
    parser.add_argument('--max-vocab', type=int, default=10000)
    parser.add_argument('--aug-start', type=float, default=0.0)
    parser.add_argument('--aug-end', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--spm-model', default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    # --------------------- Hyperparameter grid ---------------------
    param_grid = {
        "lr": [0.0001, 0.0002],
        "d_model": [128, 256],
        "nhead": [4, 8],
        "ff_dim": [256, 512],
        "num_layers": [4, 6],
        "dropout": [0.1, 0.2],
        "label_smoothing": [0.0, 0.1],
        "weight_decay": [0.0, 5e-5],
        "grad_clip": [0.0, 1.0],
    }

    # Sample 30 random combinations
    all_combos = list(itertools.product(*param_grid.values()))
    num_runs = min(30, len(all_combos))
    random.seed(42)
    sampled_combos = random.sample(all_combos, num_runs)

    print(f"Total runs: {len(sampled_combos)}")

    # --------------------- Loop over sampled hyperparameters ---------------------
    for idx, combo in enumerate(sampled_combos, 1):
        hp = {k: v for k, v in zip(param_grid.keys(), combo)}
        print(f"\n🔹 Run {idx}: {hp}")

        # Unique folder per run
        run_out_dir = Path(args.out_dir) / f"run{idx}_lr{hp['lr']}_d{hp['d_model']}_h{hp['nhead']}_l{hp['num_layers']}"
        run_out_dir.mkdir(parents=True, exist_ok=True)

        # --------------------- Dataset ---------------------
        build_vocab = True if args.spm_model is None else False
        ds = TextCommandsDataset(args.csv, text_col=args.text_col, label_col=args.label_col,
                                 build_vocab=build_vocab, max_vocab=args.max_vocab, spm_model=args.spm_model)
        label_map = ds.get_label_map()
        num_classes = len(label_map)
        print('Found classes:', label_map)

        # Stratified split
        indices = list(range(len(ds)))
        labels = [ds[i][1] for i in indices]
        train_idx, val_idx = train_test_split(indices, test_size=0.1, stratify=labels, random_state=args.seed)
        train_ds = Subset(ds, train_idx)
        val_ds = Subset(ds, val_idx)

        collate = lambda b: collate_batch(b, pad_idx=0, max_len=args.max_len)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

        # --------------------- Model ---------------------
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = TransformerClassifier(
            vocab_size=len(ds.vocab),
            d_model=hp['d_model'],
            nhead=hp['nhead'],
            num_layers=hp['num_layers'],
            dim_feedforward=hp['ff_dim'],
            num_classes=num_classes,
            pad_idx=0,
            max_len=args.max_len,
            dropout=hp['dropout']
        ).to(device)

        use_cuda = device.type == 'cuda'
        scaler = amp.GradScaler() if use_cuda else None

        opt = Adam(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])
        scheduler = CosineAnnealingLR(opt, T_max=args.epochs)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=hp['label_smoothing']) if hp['label_smoothing'] > 0.0 else nn.CrossEntropyLoss()

        # --------------------- Training ---------------------
        best_val = 0.0
        patience = 3
        no_improve = 0
        metrics_log = []

        for epoch in range(1, args.epochs + 1):
            aug_intensity = args.aug_start + (args.aug_end - args.aug_start) * (epoch - 1) / max(1, args.epochs-1)
            aug_intensity = max(0.0, min(1.0, aug_intensity))

            model.train()
            train_loss, train_acc, batches = 0.0, 0.0, 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

            for xb, yb in pbar:
                cur_seq_len = xb.size(1)
                for i in range(len(xb)):
                    text_str = ds.decode_text(xb[i].tolist())
                    lang = 'ar' if any('\u0600' <= c <= '\u06FF' for c in text_str) else 'en'
                    aug_text = augment_text(text_str, lang=lang, intensity=aug_intensity)
                    xb[i] = torch.tensor(ds.encode_text(aug_text, max_len=cur_seq_len), dtype=xb.dtype)

                xb, yb = xb.to(device), yb.to(device)

                opt.zero_grad()
                with amp.autocast(enabled=use_cuda):
                    logits = model(xb)
                    loss = loss_fn(logits, yb)

                if scaler:
                    scaler.scale(loss).backward()
                    if hp['grad_clip'] > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), hp['grad_clip'])
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    if hp['grad_clip'] > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), hp['grad_clip'])
                    opt.step()

                train_loss += loss.item()
                train_acc += accuracy(logits.detach().cpu(), yb.detach().cpu())
                batches += 1
                pbar.set_postfix(loss=train_loss/batches, acc=train_acc/batches)

            # --------------------- Validation ---------------------
            model.eval()
            val_loss, val_acc, v_batches = 0.0, 0.0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    with amp.autocast(enabled=use_cuda):
                        logits = model(xb)
                        loss = loss_fn(logits, yb)
                    val_loss += loss.item()
                    val_acc += accuracy(logits.cpu(), yb.cpu())
                    v_batches += 1

            avg_val_acc = val_acc / max(1, v_batches)
            avg_val_loss = val_loss / max(1, v_batches)
            print(f"Epoch {epoch} validation acc: {avg_val_acc:.4f}, loss: {avg_val_loss:.4f}")

            metrics_log.append({
                "epoch": epoch,
                "train_loss": train_loss / max(1, batches),
                "train_acc": train_acc / max(1, batches),
                "val_loss": avg_val_loss,
                "val_acc": avg_val_acc
            })

            if avg_val_acc > best_val:
                best_val = avg_val_acc
                no_improve = 0
                ckpt = {'model_state': model.state_dict(), 'label_map': ds.label2id}
                if args.spm_model:
                    ckpt['spm_model'] = str(args.spm_model)
                try:
                    ckpt['vocab'] = ds.vocab.itos
                except:
                    pass
                torch.save(ckpt, run_out_dir / 'best.pth')
                print("Saved best model")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("Early stopping.")
                    break

            scheduler.step()

        # Save metrics & config
        with open(run_out_dir / "metrics.json", "w") as f:
            json.dump({"hyperparams": hp, "metrics": metrics_log}, f, indent=4)
        with open(run_out_dir / "config.txt", "w") as f:
            for k, v in vars(args).items():
                f.write(f"{k}: {v}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted.")

