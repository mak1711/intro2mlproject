import os
import json
from pathlib import Path
import pandas as pd
import torch

from text_classification.spmpredict import (
    load_checkpoint,
    prepare_model_from_checkpoint,
    predict_text,
    try_load_label_map_from_txt
)

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
TEST_CSV = "data/test_set.csv"
MODELS_ROOT = "models/model_search2"
MAX_LEN = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------------------------------------

def load_metrics_json(run_dir):
    metrics_path = os.path.join(run_dir, "metrics.json")
    if not os.path.isfile(metrics_path):
        return None
    with open(metrics_path, "r") as f:
        return json.load(f)


def evaluate_model(model_path, hyperparams, df_test):
    print(f"  Loading model: {model_path}")

    cp = load_checkpoint(str(model_path), DEVICE)

    # ---------------------------------------------------
    # Ensure label_map exists
    # ---------------------------------------------------
    if "label_map" not in cp:
        lm = try_load_label_map_from_txt(Path(model_path))
        if lm is not None:
            cp["label_map"] = lm

    # ---------------------------------------------------
    # Use hyperparams from metrics.json
    # ---------------------------------------------------
    model, vocab, id2label = prepare_model_from_checkpoint(
        cp,
        max_len=MAX_LEN,
        device=DEVICE,
        d_model=hyperparams.get("d_model", 128),
        nhead=hyperparams.get("nhead", 8),
        ff_dim=hyperparams.get("ff_dim", 256),
        num_layers=hyperparams.get("num_layers", 4),
        dropout=hyperparams.get("dropout", 0.1),
    )

    correct = 0
    total = 0

    for _, row in df_test.iterrows():
        text = row["text"]
        true_label = int(row["label"])

        pred = predict_text(
            model, vocab, id2label, text,
            DEVICE, max_len=MAX_LEN, topk=1
        )

        predicted_label_name = pred[0][0]

        # convert text-label → numeric ID
        predicted_label = None
        for label_id, label_name in id2label.items():
            if label_name == predicted_label_name:
                predicted_label = int(label_id)

        if predicted_label == true_label:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def main():
    df_test = pd.read_csv(TEST_CSV)
    model_runs = sorted(os.listdir(MODELS_ROOT))

    results = []

    for run in model_runs:
        run_dir = os.path.join(MODELS_ROOT, run)
        best_pth = os.path.join(run_dir, "best.pth")

        if not os.path.isfile(best_pth):
            print(f"Skipping {run}: best.pth not found")
            continue

        metrics = load_metrics_json(run_dir)
        if metrics is None:
            print(f"Skipping {run}: metrics.json not found")
            continue

        hyperparams = metrics.get("hyperparams", {})

        print(f"\nEvaluating run: {run}")
        print(f"  Using hyperparams: {hyperparams}")

        acc = evaluate_model(best_pth, hyperparams, df_test)
        print(f"  Accuracy = {acc:.4f}\n")

        results.append((run, acc))

    # Save results
    out_csv = "model_search2_results.csv"
    pd.DataFrame(results, columns=["run", "accuracy"]).to_csv(out_csv, index=False)
    print(f"\nSaved results → {out_csv}")


if __name__ == "__main__":
    main()

