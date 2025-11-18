"""
Ensemble training script for fine-tuning pretrained models on your dataset.

Changes from original:
- Uses AdamW + parameter grouping (no weight decay for bias/LayerNorm)
- Linear warmup + decay scheduler (get_linear_schedule_with_warmup)
- Optional freezing of encoder layers kept, but embeddings are trainable by default
- Dynamic padding via a tokenizer-aware collator for improved efficiency
- Mixed-precision support and scheduler stepping per optimizer step
- Saves model + tokenizer
"""

import argparse
import random
import os
from pathlib import Path
import torch
import torch.cuda.amp as amp
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

# transformers utilities
from transformers import get_linear_schedule_with_warmup, AdamW

from text_classification.dataset import TextCommandsDataset
from text_classification.pretrained_models import get_pretrained_model, PRETRAINED_MODELS


# --------------------- Seed ---------------------
def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# --------------------- Accuracy ---------------------
def accuracy_from_logits(logits, labels):
    """
    logits: tensor (batch, num_classes)
    labels: tensor (batch,)
    """
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


# --------------------- Dataset Wrapper for Pretrained Models ---------------------
class PretrainedModelDataset(Dataset):
    """
    Dataset wrapper for pretrained models that returns raw text + label.
    Tokenization is deferred to the collator (for dynamic padding).
    """
    def __init__(self, texts: list, labels: list):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'label': int(self.labels[idx])
        }


class TokenizerCollator:
    """
    Collator that batches raw texts and uses tokenizer.batch_encode_plus
    with padding='longest' to avoid unnecessary computation.
    """
    def __init__(self, tokenizer, max_length: int = 128, return_tensors: str = 'pt'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_tensors = return_tensors

    def __call__(self, batch):
        texts = [item['text'] for item in batch]
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        enc = self.tokenizer(
            texts,
            padding=True,            # dynamic (longest) padding
            truncation=True,
            max_length=self.max_length,
            return_tensors=self.return_tensors
        )
        enc['labels'] = labels
        return enc


# --------------------- Helper Functions ---------------------
def freeze_encoder_layers(model, num_freeze_layers: int):
    """Freeze the first N encoder layers. Works for many HF-style models."""
    if num_freeze_layers <= 0:
        return

    if hasattr(model, 'bert'):
        encoder = model.bert.encoder.layer
    elif hasattr(model, 'distilbert'):
        # distilbert: distilbert.transformer.layer
        encoder = model.distilbert.transformer.layer
    elif hasattr(model, 'roberta'):
        encoder = model.roberta.encoder.layer
    elif hasattr(model, 'electra'):
        encoder = model.electra.encoder.layer
    elif hasattr(model, 'albert'):
        encoder = model.albert.encoder.albert_layer_groups[0].albert_layers
    elif hasattr(model, 'xlnet'):
        encoder = model.transformer.layer
    else:
        # Unknown model type -> skip
        return

    for i, layer in enumerate(encoder):
        if i < num_freeze_layers:
            for param in layer.parameters():
                param.requires_grad = False


def get_trainable_parameters_grouped(model, weight_decay: float):
    """
    Return parameter groups for AdamW, excluding bias and LayerNorm weights from weight decay.
    """
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "ln_f.weight"]
    params_with_decay = []
    params_without_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay):
            params_without_decay.append(param)
        else:
            params_with_decay.append(param)

    return [
        {"params": params_with_decay, "weight_decay": weight_decay},
        {"params": params_without_decay, "weight_decay": 0.0},
    ]


# --------------------- Main Training Function ---------------------
def main():
    p = argparse.ArgumentParser(description='Fine-tune pretrained models for text classification (ensemble)')
    p.add_argument('--csv', required=True, help='Path to CSV dataset')
    p.add_argument('--text-col', default='text', help='Name of text column in CSV')
    p.add_argument('--label-col', default='label', help='Name of label column in CSV')
    p.add_argument('--out-dir', default='models/ensemble', help='Output directory for trained models')
    p.add_argument('--models', nargs='+', default=['bert', 'distilbert', 'roberta'],
                   help='Which models to train. Options: ' + ', '.join(PRETRAINED_MODELS.keys()))
    p.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    p.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    p.add_argument('--lr', type=float, default=2e-5, help='Learning rate for trainable layers')
    p.add_argument('--max-len', type=int, default=128, help='Maximum sequence length')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay for optimizer')
    p.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping max norm (0 disables)')
    p.add_argument('--freeze-layers', type=int, default=0, help='Number of encoder layers to freeze (default 0)')
    p.add_argument('--unfreeze-last-n', type=int, default=None, help='Unfreeze last N layers (overrides freeze-layers)')
    p.add_argument('--freeze-embeddings', action='store_true', help='If set, freeze embeddings (not recommended)')
    p.add_argument('--warmup-ratio', type=float, default=0.1, help='Warmup ratio (fraction of total steps)')
    p.add_argument('--grad-accum', type=int, default=1, help='Gradient accumulation steps')
    args = p.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    ds = TextCommandsDataset(
        args.csv,
        text_col=args.text_col,
        label_col=args.label_col,
        build_vocab=False
    )

    label_map = ds.get_label_map()
    id2label = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)
    print(f'Found {num_classes} classes: {label_map}')

    # Stratified split
    indices = list(range(len(ds)))
    labels_for_split = [ds[i][1] for i in indices]
    train_idx, val_idx = train_test_split(
        indices, test_size=0.1, stratify=labels_for_split, random_state=args.seed
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Train each model
    for model_type in args.models:
        if model_type not in PRETRAINED_MODELS:
            print(f"Skipping unknown model type: {model_type}")
            continue

        print(f"\n{'='*60}")
        print(f"Training {model_type}...")
        print(f"{'='*60}")

        model_dir = out_dir / model_type
        model_dir.mkdir(parents=True, exist_ok=True)

        # Create model wrapper
        try:
            model_wrapper = get_pretrained_model(model_type, num_classes, device)
        except Exception as e:
            print(f"Error loading model {model_type}: {e}")
            continue

        model = model_wrapper.model
        tokenizer = model_wrapper.tokenizer

        # Determine encoder layer count for this model (for freeze/unfreeze logic)
        num_encoder_layers = 12
        if model_type == 'distilbert':
            num_encoder_layers = 6
        elif model_type in ('albert', 'xlnet', 'electra', 'mbert', 'xlmr', 'roberta', 'bert'):
            num_encoder_layers = 12
        # (If you add special models, adjust above as needed)

        # Decide freeze count
        freeze_count = 0
        if args.unfreeze_last_n is not None:
            # unfreeze-last-n overrides freeze-layers
            freeze_count = max(0, num_encoder_layers - int(args.unfreeze_last_n))
        else:
            freeze_count = max(0, int(args.freeze_layers))

        print(f"\nFine-tuning Configuration:")
        print(f"  Total encoder layers: {num_encoder_layers}")
        print(f"  Layers to freeze: {freeze_count}")
        if freeze_count > 0:
            freeze_encoder_layers(model, freeze_count)
            print(f"  Froze {freeze_count} encoder layers")

        # Optionally freeze embeddings (disabled by default)
        if args.freeze_embeddings:
            if hasattr(model, 'bert'):
                for param in model.bert.embeddings.parameters():
                    param.requires_grad = False
            elif hasattr(model, 'roberta'):
                for param in model.roberta.embeddings.parameters():
                    param.requires_grad = False
            elif hasattr(model, 'electra'):
                for param in model.electra.embeddings.parameters():
                    param.requires_grad = False
            elif hasattr(model, 'distilbert'):
                for param in model.distilbert.embeddings.parameters():
                    param.requires_grad = False
            elif hasattr(model, 'xlnet'):
                for param in model.transformer.word_embeddings.parameters():
                    param.requires_grad = False
            print("  Embeddings frozen (user requested)")

        # Prepare data
        train_texts = [ds._texts[i] for i in train_idx]
        train_labels = [ds.label2id[ds._labels[i]] for i in train_idx]
        val_texts = [ds._texts[i] for i in val_idx]
        val_labels = [ds.label2id[ds._labels[i]] for i in val_idx]

        train_dataset = PretrainedModelDataset(train_texts, train_labels)
        val_dataset = PretrainedModelDataset(val_texts, val_labels)

        collator = TokenizerCollator(tokenizer, max_length=args.max_len)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collator, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                collate_fn=collator, pin_memory=True)

        # Optimizer and scheduler
        trainable_param_groups = get_trainable_parameters_grouped(model, weight_decay=args.weight_decay)
        optimizer = AdamW(trainable_param_groups, lr=args.lr)

        num_training_steps = max(1, len(train_loader) * args.epochs // max(1, args.grad_accum))
        num_warmup_steps = int(args.warmup_ratio * num_training_steps)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        loss_fn = nn.CrossEntropyLoss()  # model outputs loss directly; keep for safety

        use_cuda = (device.type == 'cuda')
        scaler = amp.GradScaler() if use_cuda else None

        best_val_acc = 0.0
        patience = 2
        no_improve = 0

        global_step = 0

        # Training loop
        for epoch in range(1, args.epochs + 1):
            model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
            running_loss = 0.0
            running_acc = 0.0
            steps_in_epoch = 0

            optimizer.zero_grad()
            for step, batch in enumerate(pbar):
                # batch contains input_ids, attention_mask, labels (and other tokenizer fields)
                # move tensors to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)

                labels = batch.pop('labels')

                with amp.autocast(enabled=use_cuda):
                    outputs = model(**batch, labels=labels)
                    # many HF models return a ModelOutput with .loss and .logits
                    loss = outputs.loss if hasattr(outputs, 'loss') else loss_fn(outputs.logits, labels)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

                    loss_value = loss.item() if isinstance(loss, torch.Tensor) else float(loss)

                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # gradient accumulation
                if (step + 1) % args.grad_accum == 0:
                    if args.grad_clip and args.grad_clip > 0:
                        if scaler:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    scheduler.step()  # step scheduler per optimizer step (same as Trainer)
                    optimizer.zero_grad()
                    global_step += 1

                running_loss += loss_value
                running_acc += accuracy_from_logits(logits.detach(), labels.detach())
                steps_in_epoch += 1

                pbar.set_postfix({
                    'loss': running_loss / steps_in_epoch,
                    'acc': running_acc / steps_in_epoch
                })

            # Validation
            model.eval()
            val_loss = 0.0
            val_acc = 0.0
            val_steps = 0
            with torch.no_grad():
                for batch in val_loader:
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.to(device)
                    labels = batch.pop('labels')

                    with amp.autocast(enabled=use_cuda):
                        outputs = model(**batch, labels=labels)
                        loss = outputs.loss if hasattr(outputs, 'loss') else loss_fn(outputs.logits, labels)
                        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

                    loss_value = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
                    val_loss += loss_value
                    val_acc += accuracy_from_logits(logits, labels)
                    val_steps += 1

            avg_val_acc = val_acc / max(1, val_steps)
            avg_val_loss = val_loss / max(1, val_steps)
            print(f"Epoch {epoch} - Val Acc: {avg_val_acc:.4f}, Val Loss: {avg_val_loss:.4f}")

            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                no_improve = 0
                # Save model state + tokenizer
                torch.save(model.state_dict(), model_dir / 'best.pth')
                try:
                    # If model_wrapper.tokenizer has save_pretrained
                    tokenizer.save_pretrained(model_dir)
                except Exception:
                    pass
                print(f"  Saved best {model_type} model (acc: {avg_val_acc:.4f})")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stopping for {model_type}")
                    break

        # Save model info
        with open(model_dir / 'info.txt', 'w', encoding='utf-8') as f:
            f.write(f"Model: {model_type}\n")
            f.write(f"Best validation accuracy: {best_val_acc:.4f}\n")
            f.write(f"Num classes: {num_classes}\n")
            f.write(f"Class mapping:\n")
            for label_str, label_id in label_map.items():
                f.write(f"  {label_id}: {label_str}\n")

    print(f"\n{'='*60}")
    print(f"Training complete! Models saved to {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
