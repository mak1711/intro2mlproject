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
import csv
from sklearn.model_selection import train_test_split

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
    # now signature: augment_text(text, lang='en', intensity=1.0)
    text = text.strip()
    # if the caller used positional args like augment_text(text, intensity)
    # interpret that correctly (i.e., augment_text(text, 0.3))
    if isinstance(lang, (int, float)):
        intensity = float(lang)
        lang = 'en'

    # base probabilities which we scale by intensity
    base_case_p = 0.2
    base_punct_p = 0.2
    base_syn_p = 0.5

    # Random case
    if random.random() < base_case_p * intensity:
        text = text.upper() if random.random() < 0.5 else text.lower()
    # Random punctuation removal/addition
    if random.random() < base_punct_p * intensity:
        text = text.replace('.', '').replace(',', '') + random.choice(['', '.', '!'])
    # Slight paraphrasing / synonym replacement (simple examples)
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

    # apply synonym/paraphrase scaled by intensity
    key = text.lower()
    if lang == 'en' and key in synonyms_en and random.random() < base_syn_p * intensity:
        text = random.choice(synonyms_en[key])
    if lang == 'ar' and key in synonyms_ar and random.random() < base_syn_p * intensity:
        text = random.choice(synonyms_ar[key])
    return text

# --------------------- Main ---------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--text-col', default='text')
    p.add_argument('--label-col', default='label')
    p.add_argument('--out-dir', default='models/transformer_aug')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--max-len', type=int, default=32)
    p.add_argument('--max-vocab', type=int, default=10000)
    # Model complexity / architecture options
    p.add_argument('--d-model', type=int, default=256, help='hidden size / embedding dim for transformer')
    p.add_argument('--nhead', type=int, default=8, help='number of attention heads')
    p.add_argument('--ff-dim', type=int, default=512, help='transformer feedforward dimension')
    p.add_argument('--num-layers', type=int, default=4, help='number of Transformer encoder layers')
    p.add_argument('--aug-start', type=float, default=0.0, help='augmentation intensity at epoch 1 (0.0-1.0)')
    p.add_argument('--aug-end', type=float, default=1.0, help='augmentation intensity at final epoch (0.0-1.0)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dropout', type=float, default=0.2, help='dropout for transformer')
    p.add_argument('--label-smoothing', type=float, default=0.1, help='label smoothing for loss (0 to disable)')
    p.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay for optimizer')
    p.add_argument('--grad-clip', type=float, default=1.0, help='max-norm for gradient clipping (0 to disable)')
    p.add_argument('--spm-model', default=None, help='path to a pretrained SentencePiece .model to use for tokenization')
    args = p.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build dataset (if --spm-model is given, use it instead of building a word vocab)
    build_vocab = True if args.spm_model is None else False
    ds = TextCommandsDataset(args.csv, text_col=args.text_col, label_col=args.label_col, build_vocab=build_vocab, max_vocab=args.max_vocab, spm_model=args.spm_model)
    label_map = ds.get_label_map()
    num_classes = len(label_map)
    print('Found classes:', label_map)

    # Save label map
    with open(out_dir / 'label_map.txt', 'w', encoding='utf-8') as f:
        for k,v in label_map.items():
            f.write(f"{k}\t{v}\n")

    # Stratified split
    indices = list(range(len(ds)))
    labels = [ds[i][1] for i in indices]
    train_idx, val_idx = train_test_split(indices, test_size=0.1, stratify=labels, random_state=args.seed)
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    collate = lambda b: collate_batch(b, pad_idx=0, max_len=args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerClassifier(
        vocab_size=len(ds.vocab),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.ff_dim,
        num_classes=num_classes,
        pad_idx=0,
        max_len=args.max_len,
        dropout=args.dropout
    )
    model.to(device)

    use_cuda = (device.type == 'cuda')
    scaler = amp.GradScaler() if use_cuda else None

    opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(opt, T_max=args.epochs)
    if args.label_smoothing and args.label_smoothing > 0.0:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        loss_fn = nn.CrossEntropyLoss()

    best_val = 0.0
    patience = 3
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        # compute augmentation intensity based on epoch (linear schedule)
        if args.epochs > 1:
            aug_intensity = args.aug_start + (args.aug_end - args.aug_start) * (epoch - 1) / (args.epochs - 1)
        else:
            aug_intensity = args.aug_end
        # clamp
        aug_intensity = max(0.0, min(1.0, aug_intensity))
        # make available for debugging
        # print(f"Epoch {epoch} augmentation intensity: {aug_intensity:.3f}")
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        train_loss, train_acc, batches = 0.0, 0.0, 0

        for xb, yb in pbar:
            # Data augmentation
            # Apply augmentation by decoding each row, augmenting, re-encoding to the
            # current batch sequence length (to avoid size mismatch) and replacing.
            cur_seq_len = xb.size(1)
            for i in range(len(xb)):
                text_str = ds.decode_text(xb[i].tolist())
                lang = 'ar' if any('\u0600' <= c <= '\u06FF' for c in text_str) else 'en'
                aug_text = augment_text(text_str, lang=lang, intensity=aug_intensity)
                new_ids = ds.encode_text(aug_text, max_len=cur_seq_len)
                xb[i] = torch.tensor(new_ids, dtype=xb.dtype)

            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            with amp.autocast(enabled=use_cuda):
                logits = model(xb)
                loss = loss_fn(logits, yb)

            if scaler is not None:
                scaler.scale(loss).backward()
                # Gradient clipping
                if args.grad_clip and args.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip and args.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                opt.step()

            train_loss += loss.item()
            train_acc += accuracy(logits.detach().cpu(), yb.detach().cpu())
            batches += 1
            pbar.set_postfix(loss=train_loss/batches, acc=train_acc/batches)

        # Validation
        model.eval()
        val_loss, val_acc, v_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                with amp.autocast(enabled=use_cuda):
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                val_loss += loss.item()
                val_acc += accuracy(logits.cpu(), yb.cpu())
                v_batches += 1

        avg_val = val_acc / max(1, v_batches)
        print(f"Epoch {epoch} validation acc: {avg_val:.4f}, loss: {val_loss/max(1,v_batches):.4f}")

        # Save best
        if avg_val > best_val:
            best_val = avg_val
            no_improve = 0
            ckpt = {'model_state': model.state_dict(), 'label_map': ds.label2id}
            # include vocab if available (non-SPM)
            try:
                ckpt['vocab'] = ds.vocab.itos
            except Exception:
                pass
            # include spm model name if using SPM
            if args.spm_model:
                ckpt['spm_model'] = str(args.spm_model)
            torch.save(ckpt, out_dir / 'best.pth')
            print('Saved best model')
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break

        scheduler.step()

    # Save config
    with open(out_dir / "config.txt", "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
