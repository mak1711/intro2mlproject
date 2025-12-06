import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
CSV_PATH = "models/nospmsearch/test_set_results_nospm.csv"
SAVE_DIR = Path("plots")
SAVE_DIR.mkdir(exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(CSV_PATH)

# Sort by accuracy for better visualization
df_sorted = df.sort_values(by="accuracy", ascending=False)

# -----------------------------
# PLOT HISTOGRAM / BAR CHART
# -----------------------------
plt.figure(figsize=(12, 6))
plt.bar(df_sorted["run"], df_sorted["accuracy"], color="skyblue")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Accuracy")
plt.xlabel("Model Run")
plt.title("Model Accuracy Comparison")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

# -----------------------------
# SAVE HIGH-QUALITY PLOTS
# -----------------------------
plt.savefig(SAVE_DIR / "model_accuracies.png", dpi=400, bbox_inches="tight")
plt.savefig(SAVE_DIR / "model_accuracies.pdf", bbox_inches="tight")
plt.savefig(SAVE_DIR / "model_accuracies.svg", bbox_inches="tight")

plt.show()

