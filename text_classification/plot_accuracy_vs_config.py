"""Plot model accuracy vs config parameters.

Reads `eval_results.csv` (produced by evaluate_models.py), scans model folders
under `models/` (excluding `models/ensemble`), reads `config.txt` from each
folder, and produces one scatter plot per parameter showing model accuracy vs
that parameter's value.

Usage:
  PYTHONPATH=. python -m text_classification.plot_accuracy_vs_config
"""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('.').resolve()
EVAL_CSV = ROOT / 'eval_results.csv'
MODELS_DIR = ROOT / 'models'

PARAM_KEYS = [
    'epochs', 'batch_size', 'lr', 'max_len', 'max_vocab',
    'd_model', 'nhead', 'ff_dim', 'num_layers',
    'aug_start', 'aug_end', 'seed', 'dropout',
    'label_smoothing', 'weight_decay', 'grad_clip'
]


def try_parse_number(s):
    try:
        if '.' in s:
            return float(s)
        return int(s)
    except Exception:
        try:
            return float(s)
        except Exception:
            return s


def read_config(config_path: Path):
    data = {}
    if not config_path.exists():
        return None
    with config_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            k, v = line.split(':', 1)
            k = k.strip()
            v = v.strip()
            data[k] = try_parse_number(v)
    return data


def main():
    if not EVAL_CSV.exists():
        print('eval_results.csv not found — run evaluate_models.py first')
        return

    rows = []
    with EVAL_CSV.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # Filter rows: exclude ensemble entries and failures
    filtered = []
    for r in rows:
        model_folder = r['model_folder']
        mode = r.get('mode', '')
        notes = r.get('notes', '')
        if model_folder.startswith('ensemble') or model_folder.startswith('models/ensemble'):
            continue
        if mode == 'failed':
            continue
        filtered.append(r)

    records = []
    for r in filtered:
        model_folder = r['model_folder']
        # try possible paths to config.txt
        candidates = [
            MODELS_DIR / model_folder / 'config.txt',
            MODELS_DIR / model_folder / 'config.txt',
            ROOT / model_folder / 'config.txt',
            Path(model_folder) / 'config.txt'
        ]
        cfg = None
        for p in candidates:
            if p.exists():
                cfg = read_config(p)
                cfg_path = p
                break
        if cfg is None:
            # try to find config.txt under models/<folder> recursively
            cand = list(MODELS_DIR.rglob(f"{model_folder}/config.txt"))
            if cand:
                cfg = read_config(cand[0])
                cfg_path = cand[0]
        if cfg is None:
            print(f'Warning: no config.txt found for {model_folder}; skipping')
            continue
        acc = float(r.get('accuracy', 0.0))
        records.append((model_folder, cfg_path, cfg, acc))

    if not records:
        print('No model records with config found; exiting')
        return

    # Prepare combined CSV with parameter values
    out_rows = []
    for model_folder, cfg_path, cfg, acc in records:
        row = {'model_folder': model_folder, 'config_path': str(cfg_path), 'accuracy': acc}
        for k in PARAM_KEYS:
            row[k] = cfg.get(k, None)
        out_rows.append(row)

    # For each parameter, plot accuracy vs value
    for k in PARAM_KEYS:
        xs = []
        ys = []
        labels = []
        for r in out_rows:
            v = r.get(k, None)
            if v is None:
                continue
            try:
                x = float(v)
            except Exception:
                # categorical
                continue
            xs.append(x)
            ys.append(r['accuracy'])
            labels.append(r['model_folder'])

        if not xs:
            print(f'No numeric values found for param {k}; skipping plot')
            continue

        plt.figure(figsize=(8,6))
        plt.scatter(xs, ys)
        for i, txt in enumerate(labels):
            plt.text(xs[i], ys[i], txt, fontsize=8, alpha=0.8)
        plt.xlabel(k)
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy vs {k}')
        if k in ('d_model', 'params', 'ff_dim'):
            plt.xscale('log')
        plt.grid(True, linestyle='--', alpha=0.5)
        out = f'accuracy_vs_{k}.png'
        plt.tight_layout()
        plt.savefig(out)
        print('Saved', out)


if __name__ == '__main__':
    main()
