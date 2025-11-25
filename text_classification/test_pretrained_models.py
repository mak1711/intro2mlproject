import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from text_classification.pretrained_models import get_pretrained_model, PRETRAINED_MODELS

# ---------------- Dataset Wrapper ----------------
class PretrainedModelDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "label": int(self.labels[idx])
        }

class TokenizerCollator:
    def __init__(self, tokenizer, max_length=128, return_tensors='pt'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_tensors = return_tensors

    def __call__(self, batch):
        texts = [item['text'] for item in batch]
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors=self.return_tensors
        )
        enc['labels'] = labels
        return enc

# ----------------- Main Testing Function -----------------
def test_models(model_paths, test_csv_path, text_col='text', label_col='label', batch_size=16, max_len=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load test data
    df = pd.read_csv(test_csv_path)
    texts = df[text_col].tolist()
    labels = df[label_col].tolist()
    test_dataset = PretrainedModelDataset(texts, labels)

    results = {}

    for model_path_str in model_paths:
        model_path = Path(model_path_str)
        model_name = model_path.name
        print(f"\n{'='*50}\nTesting model: {model_name}\n{'='*50}")

        # XLM-R trained with Hugging Face Trainer
        if model_name.lower() == "xlmr":
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.to(device)
            model.eval()
            collator = TokenizerCollator(tokenizer, max_length=max_len)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

        # Custom-trained models with best.pth
        else:
            model_type = model_name.lower()
            if model_type not in PRETRAINED_MODELS:
                print(f"Skipping unknown model type: {model_type}")
                continue

            model_wrapper = get_pretrained_model(model_type, num_classes=len(set(labels)), device=device)
            model = model_wrapper.model
            tokenizer = model_wrapper.tokenizer

            model.load_state_dict(torch.load(model_path / 'best.pth', map_location=device))
            model.to(device)
            model.eval()

            collator = TokenizerCollator(tokenizer, max_length=max_len)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

        # ----------------- Run Inference -----------------
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                labels_batch = batch.pop('labels')
                outputs = model(**batch)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds)

        print(f"Accuracy: {acc:.4f}")
        print(f"Weighted F1: {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(report)

        results[model_name] = {
            'accuracy': acc,
            'f1': f1,
            'confusion_matrix': cm,
            'report': report
        }

    return results

# ----------------- Run Testing -----------------
if __name__ == "__main__":
    model_paths = [
        "/home/kan/ML/models/ensemble/distilbert",
        "/home/kan/ML/models/ensemble/distil-mbert",
        "/home/kan/ML/models/ensemble/mdeberta",
        "/home/kan/ML/models/ensemble/xlmr"
    ]
    test_csv_path = "/home/kan/ML/data/test_set.csv"

    results = test_models(model_paths, test_csv_path)

    print("\nSummary of all models:")
    for model_name, res in results.items():
        print(f"{model_name}: Accuracy={res['accuracy']:.4f}, Weighted F1={res['f1']:.4f}")

