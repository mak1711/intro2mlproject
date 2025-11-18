"""
Wrapper classes for pretrained transformer models from Hugging Face.
Supports English, Arabic, and multilingual models.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class PretrainedModelWrapper:
    """Base wrapper class for pretrained models with standard interface."""

    def __init__(self, model_name: str, num_classes: int, device: torch.device):
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = device

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        self.model.to(device)
        self.model.eval()

    def tokenize_and_encode(self, texts, max_length: int = 128):
        if isinstance(texts, str):
            texts = [texts]

        encoded = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].to(self.device),
            'attention_mask': encoded['attention_mask'].to(self.device)
        }

    def predict(self, texts, max_length: int = 128):
        encoded = self.tokenize_and_encode(texts, max_length)
        with torch.no_grad():
            outputs = self.model(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask']
            )
        return outputs.logits

    def get_probabilities(self, texts, max_length: int = 128):
        logits = self.predict(texts, max_length)
        return torch.softmax(logits, dim=-1)

    def get_predictions_and_confidence(self, texts, max_length: int = 128):
        probs = self.get_probabilities(texts, max_length)
        confidences, predictions = torch.max(probs, dim=-1)
        return predictions.cpu().tolist(), confidences.cpu().tolist()

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()

    def get_state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def to_device(self, device):
        self.device = device
        self.model.to(device)


# ---------------- Wrappers for individual models ----------------

class BertWrapper(PretrainedModelWrapper):
    def __init__(self, num_classes: int, device: torch.device,
                 model_name: str = 'bert-base-uncased'):
        super().__init__(model_name, num_classes, device)


class DistilBertWrapper(PretrainedModelWrapper):
    def __init__(self, num_classes: int, device: torch.device,
                 model_name: str = 'distilbert-base-uncased'):
        super().__init__(model_name, num_classes, device)


class RobertaWrapper(PretrainedModelWrapper):
    def __init__(self, num_classes: int, device: torch.device,
                 model_name: str = 'roberta-base'):
        super().__init__(model_name, num_classes, device)


class XLNetWrapper(PretrainedModelWrapper):
    def __init__(self, num_classes: int, device: torch.device,
                 model_name: str = 'xlnet-base-cased'):
        super().__init__(model_name, num_classes, device)


class AlbertWrapper(PretrainedModelWrapper):
    def __init__(self, num_classes: int, device: torch.device,
                 model_name: str = 'albert-base-v2'):
        super().__init__(model_name, num_classes, device)


class ElectraWrapper(PretrainedModelWrapper):
    def __init__(self, num_classes: int, device: torch.device,
                 model_name: str = 'google/electra-base-discriminator'):
        super().__init__(model_name, num_classes, device)


# ------------------ New multilingual wrappers ------------------

class MBertWrapper(PretrainedModelWrapper):
    """Multilingual BERT (supports Arabic + English)."""
    def __init__(self, num_classes: int, device: torch.device,
                 model_name: str = 'bert-base-multilingual-cased'):
        super().__init__(model_name, num_classes, device)


class XLMRobertaWrapper(PretrainedModelWrapper):
    """XLM-RoBERTa (supports Arabic + English)."""
    def __init__(self, num_classes: int, device: torch.device,
                 model_name: str = 'xlm-roberta-base'):
        super().__init__(model_name, num_classes, device)


class DistilMBertWrapper(PretrainedModelWrapper):
    """DistilBERT multilingual (smaller, fits in low VRAM)."""
    def __init__(self, num_classes: int, device: torch.device,
                 model_name: str = 'distilbert-base-multilingual-cased'):
        super().__init__(model_name, num_classes, device)


class MDeBertaWrapper(PretrainedModelWrapper):
    """DeBERTa multilingual small variant."""
    def __init__(self, num_classes: int, device: torch.device,
                 model_name: str = 'microsoft/deberta-base'):
        super().__init__(model_name, num_classes, device)


# ---------------- Mapping ----------------

PRETRAINED_MODELS = {
    'bert': BertWrapper,
    'distilbert': DistilBertWrapper,
    'roberta': RobertaWrapper,
    'xlnet': XLNetWrapper,
    'albert': AlbertWrapper,
    'electra': ElectraWrapper,

    # Multilingual support
    'mbert': MBertWrapper,
    'xlmr': XLMRobertaWrapper,
    'distil-mbert': DistilMBertWrapper,
    'mdeberta': MDeBertaWrapper
}


def get_pretrained_model(model_type: str, num_classes: int, device: torch.device):
    if model_type not in PRETRAINED_MODELS:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Supported types: {list(PRETRAINED_MODELS.keys())}"
        )
    wrapper_class = PRETRAINED_MODELS[model_type]
    return wrapper_class(num_classes, device)
