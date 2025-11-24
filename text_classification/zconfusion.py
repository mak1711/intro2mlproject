import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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
TEST_CSV = "/home/kan/ML/data/test_set.csv"
MODEL_PTH = "/home/kan/ML/models/model_search2/run17_lr0.0002_d256_h4_l4/best.pth"
MAX_LEN = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

# Load model
# If you have metrics.json, you can pull hyperparams from there
from pathlib import Path
import json

metrics_path = Path(MODEL_PTH).parent / "metrics.json"
with open(metrics_path) as f:
    metrics = json.load(f)
hp = metrics["hyperparams"]

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

# Prepare lists for confusion matrix
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

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(id2label.values()))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

