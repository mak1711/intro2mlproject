import os
import json
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
MODELS_ROOT = "models/model_search2"
OUT_PNG = "accuracy_plot.png"
OUT_PDF = "accuracy_plot.pdf"
# -----------------------------

def load_metrics(run_dir):
    path = os.path.join(run_dir, "metrics.json")
    if os.path.isfile(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def main():
    runs = sorted(os.listdir(MODELS_ROOT))
    plt.figure(figsize=(12, 7))

    for run in runs:
        run_dir = os.path.join(MODELS_ROOT, run)
        metrics_data = load_metrics(run_dir)

        if metrics_data is None:
            print(f"Skipping {run}, metrics.json not found.")
            continue

        metrics_list = metrics_data.get("metrics", [])
        if not metrics_list:
            print(f"Skipping {run}, no 'metrics' list found.")
            continue

        epochs = [m["epoch"] for m in metrics_list]
        train_acc = [m["train_acc"] for m in metrics_list]
        val_acc = [m["val_acc"] for m in metrics_list]

        # Plot clean lines (no markers)
        plt.plot(epochs, train_acc, linestyle='-')
        plt.plot(epochs, val_acc, linestyle='--')

    plt.title("Training and Validation Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()

    # REMOVE LEGEND COMPLETELY
    # (just do NOT call plt.legend())

    # Save high-quality output
    plt.savefig(OUT_PNG, dpi=400, bbox_inches="tight")
    plt.savefig(OUT_PDF, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()

