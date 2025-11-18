"""
Utility functions for managing ensemble models.

This module provides helper functions for:
- Creating ensemble configurations
- Loading/saving ensemble models
- Analyzing ensemble performance
- Switching between different ensemble setups
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import torch
import numpy as np
from collections import defaultdict


class EnsembleConfig:
    """Configuration for an ensemble setup."""
    
    def __init__(self):
        self.custom_model_path: Optional[Path] = None
        self.ensemble_dir: Optional[Path] = None
        self.active_models: List[str] = []
        self.model_weights: Dict[str, float] = {}
        self.custom_max_len: int = 32
        self.pretrained_max_len: int = 128
        self.use_voting_strategy: str = 'majority'  # or 'weighted'
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'custom_model_path': str(self.custom_model_path) if self.custom_model_path else None,
            'ensemble_dir': str(self.ensemble_dir) if self.ensemble_dir else None,
            'active_models': self.active_models,
            'model_weights': self.model_weights,
            'custom_max_len': self.custom_max_len,
            'pretrained_max_len': self.pretrained_max_len,
            'use_voting_strategy': self.use_voting_strategy,
        }
    
    def save(self, config_path: Path):
        """Save config to JSON file."""
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, config_path: Path) -> 'EnsembleConfig':
        """Load config from JSON file."""
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        config = cls()
        config.custom_model_path = Path(data['custom_model_path']) if data.get('custom_model_path') else None
        config.ensemble_dir = Path(data['ensemble_dir']) if data.get('ensemble_dir') else None
        config.active_models = data.get('active_models', [])
        config.model_weights = data.get('model_weights', {})
        config.custom_max_len = data.get('custom_max_len', 32)
        config.pretrained_max_len = data.get('pretrained_max_len', 128)
        config.use_voting_strategy = data.get('use_voting_strategy', 'majority')
        return config


def analyze_voting_patterns(voting_details: List[dict]) -> dict:
    """
    Analyze voting patterns across multiple predictions.
    
    Args:
        voting_details: List of voting detail dicts from ensemble predictions
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {
        'total_samples': len(voting_details),
        'consensus_rate': 0.0,  # % of samples where all models agree
        'disagreement_rate': 0.0,  # % of samples where models disagree
        'model_agreement_matrix': defaultdict(lambda: defaultdict(int)),  # How often each pair of models agrees
        'per_class_agreement': defaultdict(list),  # Agreement % per class
    }
    
    if not voting_details:
        return analysis
    
    total_consensus = 0
    
    for detail in voting_details:
        total_models = detail['total_models']
        agreement = detail['agreement_count']
        
        # Full consensus if all models agree
        if agreement == total_models:
            total_consensus += 1
        
        # Get the votes for this sample
        votes = detail['votes']
        vote_list = list(votes.items())
        
        # Build agreement matrix
        for i in range(len(vote_list)):
            for j in range(i + 1, len(vote_list)):
                model1, pred1 = vote_list[i]
                model2, pred2 = vote_list[j]
                
                if pred1 == pred2:
                    analysis['model_agreement_matrix'][model1][model2] += 1
                    analysis['model_agreement_matrix'][model2][model1] += 1
        
        # Per-class agreement
        pred_class = detail['final_prediction_label']
        agreement_pct = agreement / total_models
        analysis['per_class_agreement'][pred_class].append(agreement_pct)
    
    analysis['consensus_rate'] = total_consensus / len(voting_details) if voting_details else 0.0
    analysis['disagreement_rate'] = 1.0 - analysis['consensus_rate']
    
    # Average per-class agreement
    analysis['avg_per_class_agreement'] = {}
    for pred_class, agreements in analysis['per_class_agreement'].items():
        analysis['avg_per_class_agreement'][pred_class] = np.mean(agreements)
    
    return analysis


def print_analysis(analysis: dict):
    """Pretty-print voting analysis."""
    print("\n" + "="*60)
    print("VOTING ANALYSIS")
    print("="*60)
    print(f"Total samples: {analysis['total_samples']}")
    print(f"Consensus rate: {analysis['consensus_rate']:.1%}")
    print(f"Disagreement rate: {analysis['disagreement_rate']:.1%}")
    
    if analysis['avg_per_class_agreement']:
        print("\nAverage agreement by class:")
        for pred_class in sorted(analysis['avg_per_class_agreement'].keys()):
            agreement = analysis['avg_per_class_agreement'][pred_class]
            print(f"  {pred_class:20s}: {agreement:.1%}")
    
    if analysis['model_agreement_matrix']:
        print("\nModel agreement matrix:")
        models = sorted(set().union(*[set(d.keys()) for d in analysis['model_agreement_matrix'].values()]))
        
        # Header
        print("  " + " ".join(f"{m:10s}" for m in models))
        
        # Rows
        for m1 in models:
            row = [f"{m1:10s}"]
            for m2 in models:
                if m1 == m2:
                    row.append("-" * 10)
                else:
                    count = analysis['model_agreement_matrix'][m1][m2]
                    row.append(f"{count:10d}")
            print("".join(row))
    
    print("="*60 + "\n")


def get_model_disagreements(voting_details: List[dict]) -> List[dict]:
    """
    Extract samples where models disagreed.
    
    Args:
        voting_details: List of voting detail dicts
        
    Returns:
        List of disagreement records with sample text and conflicting votes
    """
    disagreements = []
    
    for detail in voting_details:
        if detail['agreement_count'] < detail['total_models']:
            disagreements.append({
                'text': detail['text'],
                'predicted_label': detail['final_prediction_label'],
                'votes': detail['votes'],
                'vote_counts': detail['vote_counts'],
                'agreement_count': detail['agreement_count'],
                'total_models': detail['total_models'],
            })
    
    return disagreements


def print_disagreements(disagreements: List[dict], max_show: int = 10):
    """Pretty-print model disagreements."""
    if not disagreements:
        print("No disagreements found!")
        return
    
    print("\n" + "="*60)
    print(f"MODEL DISAGREEMENTS ({len(disagreements)} total, showing {min(max_show, len(disagreements))})")
    print("="*60)
    
    for i, disag in enumerate(disagreements[:max_show]):
        print(f"\n{i+1}. {disag['text']}")
        print(f"   Predicted: {disag['predicted_label']}")
        print(f"   Agreement: {disag['agreement_count']}/{disag['total_models']}")
        print(f"   Votes:")
        for model, vote in disag['votes'].items():
            print(f"     {model:15s} → {vote}")
    
    if len(disagreements) > max_show:
        print(f"\n... and {len(disagreements) - max_show} more")
    
    print("="*60 + "\n")


def compute_ensemble_diversity_score(voting_details: List[dict]) -> float:
    """
    Compute ensemble diversity score (0-1).
    
    Higher values indicate more diverse predictions across models.
    A perfectly diverse ensemble would have a score closer to 1.0.
    
    Args:
        voting_details: List of voting detail dicts
        
    Returns:
        Diversity score between 0 and 1
    """
    if not voting_details:
        return 0.0
    
    # Count unique vote distributions
    unique_distributions = set()
    for detail in voting_details:
        vote_tuple = tuple(sorted(detail['vote_counts'].items()))
        unique_distributions.add(vote_tuple)
    
    # More unique distributions = higher diversity
    diversity = len(unique_distributions) / len(voting_details)
    
    return diversity


class EnsembleStats:
    """Compute and track ensemble statistics."""
    
    def __init__(self):
        self.predictions: List[str] = []
        self.confidences: List[float] = []
        self.labels: Optional[List[str]] = None
    
    def add_batch(self, predictions: List[str], confidences: List[float], labels: Optional[List[str]] = None):
        """Add a batch of predictions."""
        self.predictions.extend(predictions)
        self.confidences.extend(confidences)
        if labels:
            if self.labels is None:
                self.labels = []
            self.labels.extend(labels)
    
    def accuracy(self) -> float:
        """Compute accuracy if labels are available."""
        if self.labels is None:
            return None
        return sum(p == l for p, l in zip(self.predictions, self.labels)) / len(self.predictions)
    
    def avg_confidence(self) -> float:
        """Average confidence across predictions."""
        return np.mean(self.confidences) if self.confidences else 0.0
    
    def confidence_histogram(self, bins: int = 10) -> dict:
        """Get histogram of confidence scores."""
        if not self.confidences:
            return {}
        
        counts, edges = np.histogram(self.confidences, bins=bins, range=(0, 1))
        return {
            'bins': [f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(len(edges)-1)],
            'counts': counts.tolist(),
        }
    
    def to_dict(self) -> dict:
        """Get statistics as dictionary."""
        stats = {
            'num_predictions': len(self.predictions),
            'avg_confidence': self.avg_confidence(),
            'confidence_histogram': self.confidence_histogram(),
        }
        
        if self.labels:
            stats['accuracy'] = self.accuracy()
        
        return stats
