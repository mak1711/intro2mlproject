import os
import json
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
MODELS_ROOT = "/home/kan/ML/models/model_search2"   # folder with run1, run2, run3...
# -----------------------------

def load_metrics(run_dir):
    """Load metrics.json if it exists."""
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

        # Plot training accuracy (solid)
        plt.plot(epochs, train_acc, marker='o', label=f"{run} train")
        # Plot validation accuracy (dashed)
        plt.plot(epochs, val_acc, marker='x', linestyle='--', label=f"{run} val")

    plt.title("Training and Validation Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

