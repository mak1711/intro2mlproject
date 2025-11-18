"""
Ensemble classifier that combines multiple models with majority voting.

This module implements an ensemble that:
1. Uses your custom Transformer model
2. Uses multiple pretrained transformer models (BERT, DistilBERT, RoBERTa, etc.)
3. Takes predictions from all models
4. Uses majority voting to select the final prediction
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from collections import Counter
from pathlib import Path
import json

from text_classification.model import TransformerClassifier
from text_classification.pretrained_models import PretrainedModelWrapper, get_pretrained_model


class EnsembleClassifier:
    """
    Ensemble classifier combining custom model with multiple pretrained models.
    
    Uses majority voting to select the final prediction from all models.
    """
    
    def __init__(self, 
                 custom_model: TransformerClassifier,
                 custom_vocab,
                 custom_id2label: Dict[int, str],
                 pretrained_models: Optional[Dict[str, PretrainedModelWrapper]] = None,
                 device: torch.device = None):
        """
        Initialize ensemble classifier.
        
        Args:
            custom_model: Your custom TransformerClassifier
            custom_vocab: Vocabulary object for custom model
            custom_id2label: Mapping from class index to label string
            pretrained_models: Dict of model_name -> PretrainedModelWrapper instances
            device: Torch device (cpu or cuda)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.custom_model = custom_model
        self.custom_vocab = custom_vocab
        self.custom_id2label = custom_id2label
        self.num_classes = len(custom_id2label)
        
        # Create label2id from id2label
        self.label2id = {v: k for k, v in custom_id2label.items()}
        
        # Pretrained models
        self.pretrained_models = pretrained_models or {}
        
        # Track which models are available
        self.model_names = ['custom'] + list(self.pretrained_models.keys())
        
    def add_pretrained_model(self, model_name: str, model: PretrainedModelWrapper):
        """Add a pretrained model to the ensemble."""
        self.pretrained_models[model_name] = model
        self.model_names = ['custom'] + list(self.pretrained_models.keys())
    
    def remove_pretrained_model(self, model_name: str):
        """Remove a pretrained model from the ensemble."""
        if model_name in self.pretrained_models:
            del self.pretrained_models[model_name]
            self.model_names = ['custom'] + list(self.pretrained_models.keys())
    
    def predict_custom(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Get predictions from custom model.
        
        Args:
            ids: Token IDs tensor of shape (batch_size, seq_len)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        with torch.no_grad():
            logits = self.custom_model(ids)
        return logits
    
    def predict_pretrained(self, texts: List[str], max_length: int = 128) -> Dict[str, torch.Tensor]:
        """
        Get predictions from all pretrained models.
        
        Args:
            texts: List of text strings
            max_length: Max token sequence length for pretrained models
            
        Returns:
            Dict mapping model_name -> logits tensor of shape (batch_size, num_classes)
        """
        results = {}
        for model_name, model in self.pretrained_models.items():
            results[model_name] = model.predict(texts, max_length)
        return results
    
    def predict_ensemble(self, 
                        texts: List[str],
                        use_custom: bool = True,
                        use_pretrained: bool = True,
                        custom_max_len: int = 32,
                        pretrained_max_len: int = 128) -> Tuple[List[str], Dict]:
        """
        Get ensemble predictions using majority voting.
        
        Args:
            texts: List of text strings (or can be single string)
            use_custom: Whether to include custom model
            use_pretrained: Whether to include pretrained models
            custom_max_len: Max sequence length for custom model
            pretrained_max_len: Max sequence length for pretrained models
            
        Returns:
            Tuple of:
            - predictions: List of predicted label strings
            - details: Dict with voting details and per-model predictions
        """
        if isinstance(texts, str):
            texts = [texts]
        
        batch_size = len(texts)
        all_predictions = []  # List of lists: predictions per model per sample
        all_logits = {}  # model_name -> logits
        
        # Get predictions from custom model
        if use_custom:
            from text_classification.utils import simple_tokenize
            
            custom_ids_list = []
            for text in texts:
                toks = simple_tokenize(text)
                ids = self.custom_vocab.encode(toks)[:custom_max_len]
                if len(ids) < custom_max_len:
                    ids = ids + [0] * (custom_max_len - len(ids))
                custom_ids_list.append(ids)
            
            custom_ids = torch.tensor(custom_ids_list, dtype=torch.long, device=self.device)
            custom_logits = self.predict_custom(custom_ids)
            all_logits['custom'] = custom_logits
            
            custom_preds = custom_logits.argmax(dim=1).cpu().tolist()
            all_predictions.append(custom_preds)
        
        # Get predictions from pretrained models
        if use_pretrained:
            pretrained_logits = self.predict_pretrained(texts, pretrained_max_len)
            for model_name, logits in pretrained_logits.items():
                all_logits[model_name] = logits
                preds = logits.argmax(dim=1).cpu().tolist()
                all_predictions.append(preds)
        
        # Perform majority voting
        final_predictions = []
        voting_details = []
        
        for sample_idx in range(batch_size):
            # Get predictions for this sample from all models
            votes = [all_predictions[model_idx][sample_idx] 
                    for model_idx in range(len(all_predictions))]
            
            # Majority voting
            vote_counts = Counter(votes)
            most_common_class, vote_count = vote_counts.most_common(1)[0]
            
            final_predictions.append(self.custom_id2label[most_common_class])
            
            voting_details.append({
                'text': texts[sample_idx],
                'votes': {self.model_names[i]: votes[i] for i in range(len(votes))},
                'vote_counts': {self.custom_id2label[k]: v for k, v in vote_counts.items()},
                'final_prediction': most_common_class,
                'final_prediction_label': self.custom_id2label[most_common_class],
                'agreement_count': vote_count,
                'total_models': len(votes),
            })
        
        details = {
            'voting_details': voting_details,
            'all_logits': all_logits,
            'active_models': self.model_names,
        }
        
        return final_predictions, details
    
    def predict_with_probabilities(self,
                                  texts: List[str],
                                  use_custom: bool = True,
                                  use_pretrained: bool = True,
                                  custom_max_len: int = 32,
                                  pretrained_max_len: int = 128) -> Tuple[List[str], List[float], Dict]:
        """
        Get ensemble predictions with confidence scores.
        
        Confidence is computed as the proportion of models agreeing with the final prediction.
        
        Args:
            texts: List of text strings
            use_custom: Whether to include custom model
            use_pretrained: Whether to include pretrained models
            custom_max_len: Max sequence length for custom model
            pretrained_max_len: Max sequence length for pretrained models
            
        Returns:
            Tuple of:
            - predictions: List of predicted label strings
            - confidences: List of confidence scores (0-1)
            - details: Dict with voting details
        """
        predictions, details = self.predict_ensemble(
            texts, use_custom, use_pretrained, custom_max_len, pretrained_max_len
        )
        
        confidences = []
        for vote_detail in details['voting_details']:
            total = vote_detail['total_models']
            agreement = vote_detail['agreement_count']
            confidence = agreement / total
            confidences.append(confidence)
        
        return predictions, confidences, details
    
    def save_models(self, save_dir: Path):
        """
        Save custom and pretrained models.
        
        Args:
            save_dir: Directory to save models to
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save custom model
        custom_state = {
            'model_state': self.custom_model.state_dict(),
            'id2label': self.custom_id2label,
        }
        torch.save(custom_state, save_dir / 'custom_model.pth')
        
        # Save each pretrained model
        for model_name, model in self.pretrained_models.items():
            state_dict = model.get_state_dict()
            torch.save(state_dict, save_dir / f'{model_name}_model.pth')
        
        # Save ensemble metadata
        metadata = {
            'num_classes': self.num_classes,
            'label2id': self.label2id,
            'id2label': {str(k): v for k, v in self.custom_id2label.items()},
            'model_names': self.model_names,
            'pretrained_model_names': list(self.pretrained_models.keys()),
        }
        with open(save_dir / 'ensemble_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def eval(self):
        """Set all models to evaluation mode."""
        self.custom_model.eval()
        for model in self.pretrained_models.values():
            model.eval_mode()
    
    def train(self):
        """Set all models to training mode."""
        self.custom_model.train()
        for model in self.pretrained_models.values():
            model.train_mode()
