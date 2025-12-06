import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
import matplotlib.pyplot as plt

from text_classification.spmpredict import (
    load_checkpoint,
    prepare_model_from_checkpoint,
    predict_text,
    try_load_label_map_from_txt
)

# -----------------------------
# CONFIG
# -----------------------------
TEST_CSV = "data/test_set.csv"
MODEL_PTH = "models/spmfinal/best.pth"
MAX_LEN = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = "/home/kan/ML/eval_plots"   # folder to save all plots
import os
os.makedirs(SAVE_DIR, exist_ok=True)
# -----------------------------

# Load test set
df_test = pd.read_csv(TEST_CSV)

# Load checkpoint
cp = load_checkpoint(MODEL_PTH, DEVICE)

# Ensure label_map exists
if "label_map" not in cp:
    lm = try_load_label_map_from_txt(MODEL_PTH)
    if lm is not None:
        cp["label_map"] = lm

# Load model hyperparameters
from pathlib import Path
import json

metrics_path = "/home/kan/ML/models/model_search2/run17_lr0.0002_d256_h4_l4/metrics.json"
with open(metrics_path) as f:
    metrics = json.load(f)
hp = metrics["hyperparams"]

# Prepare model
model, vocab, id2label = prepare_model_from_checkpoint(
    cp,
    max_len=MAX_LEN,
    device=DEVICE,
    d_model=hp["d_model"],
    nhead=hp["nhead"],
    ff_dim=hp["ff_dim"],
    num_layers=hp["num_layers"],
    dropout=hp["dropout"],
)
model.eval()

# Prepare lists
y_true = []
y_pred = []

for _, row in df_test.iterrows():
    text = row["text"]
    true_label = int(row["label"])
    y_true.append(true_label)

    pred = predict_text(model, vocab, id2label, text, DEVICE, max_len=MAX_LEN, topk=1)
    pred_label_name = pred[0][0]

    # Convert predicted label name → numeric ID
    for k, v in id2label.items():
        if v == pred_label_name:
            y_pred.append(int(k))
            break

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(id2label.values()))
fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax_cm)
ax_cm.set_title("Confusion Matrix")

# Save high-quality
cm_png = os.path.join(SAVE_DIR, "confusion_matrix.png")
cm_pdf = os.path.join(SAVE_DIR, "confusion_matrix.pdf")
fig_cm.savefig(cm_png, dpi=400, bbox_inches="tight")
fig_cm.savefig(cm_pdf, bbox_inches="tight")

plt.show()

# -----------------------------
# Accuracy
# -----------------------------
acc = accuracy_score(y_true, y_pred)
print(f"Accuracy: {acc:.4f}")

# -----------------------------
# Classification metrics table
# -----------------------------
report_dict = classification_report(
    y_true, y_pred, target_names=list(id2label.values()), output_dict=True
)
report_df = pd.DataFrame(report_dict).transpose()

# Per-class metrics only
class_metrics = report_df.iloc[:-3, :].round(2)

# -----------------------------
# Plot metrics table
# -----------------------------
fig_tbl, ax_tbl = plt.subplots(figsize=(12, len(class_metrics)*0.5 + 1))
ax_tbl.axis('off')
table = ax_tbl.table(
    cellText=class_metrics.values,
    colLabels=class_metrics.columns,
    rowLabels=class_metrics.index,
    cellLoc='center',
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
plt.title("Per-Class Metrics Table (Precision, Recall, F1, Support)")

# Save high-quality
tbl_png = os.path.join(SAVE_DIR, "metrics_table.png")
tbl_pdf = os.path.join(SAVE_DIR, "metrics_table.pdf")
fig_tbl.savefig(tbl_png, dpi=400, bbox_inches="tight")
fig_tbl.savefig(tbl_pdf, bbox_inches="tight")

plt.show()

