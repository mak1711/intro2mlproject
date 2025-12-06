"""
Ensemble Search for Fine-tuning Pretrained Models
Automatically runs multiple freeze configurations for selected models.
Shows progress and saves per-epoch metrics.
"""

import os
from pathlib import Path
import subprocess
import json

# --------------------- Settings ---------------------
CSV_PATH = "data/updated_data.csv"  # path to your dataset
OUT_DIR = "models/freeze_experiments"
EPOCHS = 7
BATCH_SIZE = 16
MAX_LEN = 128
LR = 2e-5

# Models to run
MODELS = ["distilbert", "distil-mbert", "mdeberta"]

# Freezing configurations (example choices per model)
FREEZE_CONFIGS = {
    "distilbert": [2, 3, 4, 5, 6],
    "distil-mbert": [2, 3, 4, 5, 6],
    "mdeberta": [4, 6, 8, 10, 12],
}

# Limit to 10 runs per model
for model in MODELS:
    freezes = FREEZE_CONFIGS.get(model, [0])[:10]

    for freeze_layers in freezes:
        save_dir = Path(OUT_DIR) / model / f"freeze_{freeze_layers}"
        save_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*60)
        print(f"Running {model} with freeze_layers={freeze_layers}")
        print(f"Saving to: {save_dir}")
        print("="*60)

        # Build the command
        cmd = [
            "python3", "-m", "text_classification.train_ensemble",
            "--csv", CSV_PATH,
            "--epochs", str(EPOCHS),
            "--batch-size", str(BATCH_SIZE),
            "--max-len", str(MAX_LEN),
            "--lr", str(LR),
            "--out-dir", str(save_dir),
            "--models", model,
            "--freeze-layers", str(freeze_layers)
        ]

        # Run training and show live output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        metrics = {"train": [], "val": []}

        # Capture live output
        for line in process.stdout:
            print(line, end='')  # print live progress

            # Optional: parse lines to extract per-epoch metrics
            # Example: "Epoch 1 - Val Acc: 0.8732, Val Loss: 0.4211"
            if "Epoch" in line and "Val Acc" in line:
                try:
                    epoch_num = int(line.split("Epoch")[1].split("-")[0].strip())
                    val_acc = float(line.split("Val Acc:")[1].split(",")[0].strip())
                    val_loss = float(line.split("Val Loss:")[1].strip())
                    metrics["val"].append({
                        "epoch": epoch_num,
                        "val_acc": val_acc,
                        "val_loss": val_loss
                    })
                except Exception:
                    pass

        process.wait()
        if process.returncode != 0:
            print(f"⚠️  Training failed for {model} with freeze_layers={freeze_layers}")
        else:
            print(f"✅ Finished {model} freeze_layers={freeze_layers}")
            # Save metrics
            metrics_file = save_dir / "metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=4)

