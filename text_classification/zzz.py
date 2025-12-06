import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
SAVE_DIR = Path("plots")
SAVE_DIR.mkdir(exist_ok=True)

# -----------------------------
# DATA: Extracted from your logs
# -----------------------------
epochs = list(range(1, 14))  # epochs 1 to 13

# Training metrics
train_acc = [0.643, 0.989, 0.993, 0.993, 0.996, 0.997, 0.997, 0.997, 0.997, 0.997, 0.998, 0.998, 0.999]
train_loss = [1.33, 0.557, 0.516, 0.508, 0.5, 0.497, 0.495, 0.494, 0.493, 0.492, 0.491, 0.489, 0.489]

# Validation metrics
val_acc = [0.8753, 0.9112, 0.9242, 0.9782, 0.9825, 0.9844, 0.9903, 0.9894, 0.9844, 0.9913, 0.9913, 0.9913, 0.9913]
val_loss = [0.8951, 0.6676, 0.6213, 0.5619, 0.5515, 0.5397, 0.5380, 0.5461, 0.5368, 0.5246, 0.5243, 0.5266, 0.5299]

# -----------------------------
# ACCURACY PLOT
# -----------------------------
plt.figure(figsize=(10,6))
plt.plot(epochs, train_acc, marker='o', linestyle='-', label="Train Accuracy")
plt.plot(epochs, val_acc, marker='x', linestyle='--', label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy Over Epochs")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(SAVE_DIR / "accuracy_over_epochs.png", dpi=400, bbox_inches="tight")
plt.savefig(SAVE_DIR / "accuracy_over_epochs.pdf", bbox_inches="tight")
plt.savefig(SAVE_DIR / "accuracy_over_epochs.svg", bbox_inches="tight")
plt.close()

# -----------------------------
# LOSS PLOT
# -----------------------------
plt.figure(figsize=(10,6))
plt.plot(epochs, train_loss, marker='o', linestyle='-', label="Train Loss")
plt.plot(epochs, val_loss, marker='x', linestyle='--', label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(SAVE_DIR / "loss_over_epochs.png", dpi=400, bbox_inches="tight")
plt.savefig(SAVE_DIR / "loss_over_epochs.pdf", bbox_inches="tight")
plt.savefig(SAVE_DIR / "loss_over_epochs.svg", bbox_inches="tight")
plt.close()

