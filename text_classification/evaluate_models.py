"""Evaluate saved models on a CSV dataset and plot accuracy vs model and vs parameter count.

Saves two PNG files into the current working directory:
 - accuracy_by_model.png
 - accuracy_vs_params.png

Usage:
  PYTHONPATH=. python -m text_classification.evaluate_models --csv data/robot_commands_no_mixed_language.csv \
      --ensemble-dir models/ensemble --custom-model models/mild_reg_2ep/best.pth
"""
import argparse
from pathlib import Path
import torch
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from text_classification.dataset import TextCommandsDataset
from text_classification.pretrained_models import get_pretrained_model, PRETRAINED_MODELS
from text_classification.predict_ensemble import load_custom_model
import csv


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def evaluate_pretrained(model_wrapper, texts, labels, batch_size=64, max_len=128, device=None):
    n = len(texts)
    correct = 0
    total = 0
    model_wrapper.to_device(device)
    model_wrapper.eval_mode()
    for i in range(0, n, batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        with torch.no_grad():
            probs = model_wrapper.get_probabilities(batch_texts, max_length=max_len)
            preds = torch.argmax(probs, dim=-1).cpu().numpy()
        for p, y in zip(preds, batch_labels):
            if int(p) == int(y):
                correct += 1
            total += 1
    return correct / max(1, total)


def evaluate_custom(model, vocab, texts, labels, batch_size=64, max_len=32, device=None):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        ids_batch = []
        for t in batch_texts:
            # simple encoding using vocab and simple_tokenize
            from text_classification.utils import simple_tokenize
            toks = simple_tokenize(t)
            enc = vocab.encode(toks)
            if len(enc) < max_len:
                enc = enc + [0] * (max_len - len(enc))
            else:
                enc = enc[:max_len]
            ids_batch.append(enc)
        x = torch.tensor(ids_batch, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
        for p, y in zip(preds, batch_labels):
            if int(p) == int(y):
                correct += 1
            total += 1
    return correct / max(1, total)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--ensemble-dir', default='models/ensemble')
    p.add_argument('--custom-model', default=None)
    p.add_argument('--device', default=None)
    p.add_argument('--batch-size', type=int, default=64)
    args = p.parse_args()

    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    ds = TextCommandsDataset(args.csv)
    texts = ds._texts
    labels = [ds.label2id[l] for l in ds._labels]
    num_classes = len(ds.labels)
    print(f'Dataset: {len(texts)} samples, {num_classes} classes')

    results = []

    # Scan all model folders under models_dir for best.pth and evaluate each
    models_root = Path(args.ensemble_dir)
    print(f'Scanning models root: {models_root.resolve()}')
    best_paths = []
    # find all best.pth files one level deep (and deeper)
    for p in models_root.rglob('best.pth'):
        # skip cached HF model folders
        if 'hf_cache' in str(p):
            continue
        best_paths.append(p)

    # also include sibling model folders at models_root.parent (in case user passed models/ensemble)
    parent_root = models_root.parent
    if parent_root != models_root:
        for p in parent_root.rglob('best.pth'):
            if 'hf_cache' in str(p):
                continue
            if p not in best_paths:
                best_paths.append(p)

    # Deduplicate and sort
    best_paths = sorted(set(best_paths))
    print(f'Found {len(best_paths)} checkpoints to evaluate')

    for ck in best_paths:
        sub = ck.parent
        model_folder = str(sub.relative_to(models_root.parent)) if models_root.parent in sub.parents or sub == models_root.parent else str(sub)
        model_type = sub.name
        print('Evaluating model folder:', sub)

        # Try pretrained wrapper if the folder name matches known pretrained keys
        if model_type in PRETRAINED_MODELS:
            try:
                wrapper = get_pretrained_model(model_type, num_classes, device)
                wrapper.load_state_dict(torch.load(ck, map_location=device))
                wrapper.eval_mode()
                acc = evaluate_pretrained(wrapper, texts, labels, batch_size=args.batch_size, max_len=128, device=device)
                params = count_parameters(wrapper.model)
                results.append((model_folder, 'pretrained', model_type, acc, params, ''))
                print(f'  pretrained acc={acc:.4f}, params={params}')
                continue
            except Exception as e:
                print('  pretrained wrapper failed, will try custom load:', e)

        # Fallback: try to load as a custom Transformer checkpoint
        try:
            model, vocab, id2label = load_custom_model(ck, device)
            try:
                max_len = model.pos.pe.shape[1]
            except Exception:
                max_len = 32
            acc = evaluate_custom(model, vocab, texts, labels, batch_size=args.batch_size, max_len=max_len, device=device)
            params = count_parameters(model)
            results.append((model_folder, 'custom', model_type, acc, params, ''))
            print(f'  custom acc={acc:.4f}, params={params}')
        except Exception as e:
            print('  Failed to evaluate checkpoint as custom model:', e)
            results.append((model_folder, 'failed', model_type, 0.0, 0, str(e)))

    # Evaluate custom model
    if args.custom_model:
        ck = Path(args.custom_model)
        if ck.exists():
            print('Evaluating custom model:', ck)
            model, vocab, id2label = load_custom_model(ck, device)
            # infer pos length if possible
            try:
                max_len = model.pos.pe.shape[1]
            except Exception:
                max_len = 32
            acc = evaluate_custom(model, vocab, texts, labels, batch_size=args.batch_size, max_len=max_len, device=device)
            params = count_parameters(model)
            # use a consistent result tuple: (model_folder, mode, model_type, accuracy, params, notes)
            results.append((str(ck.parent), 'custom', ck.parent.name, acc, params, ''))
            print(f'  acc={acc:.4f}, params={params}')

    # Save results to CSV and plot
    out_csv = Path('eval_results.csv')
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['model_folder', 'mode', 'model_type', 'accuracy', 'params', 'notes'])
        for r in results:
            writer.writerow(r)
    print(f'Saved numeric results to {out_csv}')

    # Save results and plot
    if not results:
        print('No models evaluated.')
        return

    names = [str(r[0]) for r in results]
    accs = [float(r[3]) for r in results]
    params = [int(r[4]) for r in results]

    # Bar plot accuracy by model
    plt.figure(figsize=(10,6))
    x = np.arange(len(names))
    plt.bar(x, accs, color='C0')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.ylabel('Accuracy')
    plt.title('Model accuracy on dataset')
    plt.tight_layout()
    plt.savefig('accuracy_by_model.png')
    print('Saved accuracy_by_model.png')

    # Scatter plot accuracy vs params
    plt.figure(figsize=(8,6))
    plt.scatter(params, accs)
    for i, name in enumerate(names):
        plt.text(params[i], accs[i], name)
    plt.xscale('log')
    plt.xlabel('Parameter count (log scale)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Parameter count')
    plt.tight_layout()
    plt.savefig('accuracy_vs_params.png')
    print('Saved accuracy_vs_params.png')


if __name__ == '__main__':
    main()
