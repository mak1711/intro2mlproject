import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
MODELS_ROOT = "models/nospm_search"
RESULT_CSV = "model_search2_results_nospm.csv"

HYPERPARAMS = [
    "epochs", "batch_size", "lr", "max_len", "max_vocab", "d_model",
    "nhead", "ff_dim", "num_layers", "aug_start", "aug_end",
    "seed", "dropout", "label_smoothing", "weight_decay", "grad_clip"
]

# -----------------------------
# LOAD ACCURACY CSV
# -----------------------------
results_df = pd.read_csv(RESULT_CSV)

# -----------------------------
# LOAD CONFIG.TXT
# -----------------------------
def read_config_txt(path):
    params = {}
    if not os.path.isfile(path):
        return params
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                try:
                    if "." in val:
                        val = float(val)
                    else:
                        val = int(val)
                except ValueError:
                    pass
                params[key] = val
    return params

records = []
for _, row in results_df.iterrows():
    run = row["run"]
    acc = row["accuracy"]
    run_dir = os.path.join(MODELS_ROOT, run)
    config_path = os.path.join(run_dir, "config.txt")
    hyperparams = read_config_txt(config_path)
    record = {"run": run, "accuracy": acc}
    record.update(hyperparams)
    records.append(record)

df = pd.DataFrame(records)

# -----------------------------
# PLOTTING
# -----------------------------
for hp in HYPERPARAMS:
    if hp in df.columns:
        # sort by hyperparameter value for line plot
        df_sorted = df.sort_values(by=hp)
        plt.figure(figsize=(6,4))
        plt.plot(df_sorted[hp], df_sorted["accuracy"], marker='o', linestyle='-')
        plt.xlabel(hp)
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy vs {hp}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

