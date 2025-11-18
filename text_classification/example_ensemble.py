#!/usr/bin/env python3
"""
Example script demonstrating the ensemble learning workflow.

This script shows:
1. Loading the custom model
2. Loading fine-tuned pretrained models
3. Using the ensemble with majority voting
4. Analyzing voting patterns
"""

import sys
from pathlib import Path

# Add project root to path
proj_root = str(Path(__file__).resolve().parent.parent)
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from text_classification.model import TransformerClassifier
from text_classification.dataset import Vocab, TextCommandsDataset
from text_classification.pretrained_models import get_pretrained_model, PRETRAINED_MODELS
from text_classification.ensemble import EnsembleClassifier
from text_classification.ensemble_utils import (
    analyze_voting_patterns,
    print_analysis,
    get_model_disagreements,
    print_disagreements,
    compute_ensemble_diversity_score,
)
import torch


def load_custom_model(model_path: Path, device: torch.device):
    """Load the custom transformer model."""
    print(f"Loading custom model from {model_path}...")
    cp = torch.load(model_path, map_location=device)
    
    label_map = cp.get('label_map')
    id2label = {int(v): k for k, v in label_map.items()}
    
    itos = cp.get('vocab')
    vocab = Vocab(tokens=[])
    vocab.itos = itos
    vocab.stoi = {t: i for i, t in enumerate(itos)}
    
    model = TransformerClassifier(vocab_size=len(vocab), num_classes=len(id2label), pad_idx=0)
    model.load_state_dict(cp['model_state'])
    model.to(device)
    model.eval()
    
    print(f"  Classes: {id2label}")
    return model, vocab, id2label


def load_pretrained_models(ensemble_dir: Path, device: torch.device, num_classes: int):
    """Load all available pretrained models."""
    print(f"\nLoading pretrained models from {ensemble_dir}...")
    pretrained_models = {}
    
    for model_type in PRETRAINED_MODELS.keys():
        model_path = ensemble_dir / model_type / 'best.pth'
        
        if not model_path.exists():
            print(f"  {model_type:15s} - Not found, skipping")
            continue
        
        try:
            print(f"  {model_type:15s} - Loading...", end=' ')
            model_wrapper = get_pretrained_model(model_type, num_classes, device)
            model_wrapper.load_state_dict(torch.load(model_path, map_location=device))
            model_wrapper.eval_mode()
            pretrained_models[model_type] = model_wrapper
            print("✓")
        except Exception as e:
            print(f"✗ ({e})")
    
    return pretrained_models


def example_basic_prediction():
    """Basic prediction example."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Ensemble Prediction")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load models
    custom_model_path = Path('models/test_run/best.pth')
    ensemble_dir = Path('models/ensemble')
    
    if not custom_model_path.exists():
        print(f"ERROR: Custom model not found at {custom_model_path}")
        print("Please train your custom model first:")
        print("  python -m text_classification.train --csv data/your_data.csv")
        return
    
    custom_model, vocab, id2label = load_custom_model(custom_model_path, device)
    pretrained_models = load_pretrained_models(ensemble_dir, device, len(id2label))
    
    if not pretrained_models:
        print("\nWARNING: No pretrained models loaded. Train them first:")
        print("  python -m text_classification.train_ensemble --csv data/your_data.csv")
    
    # Create ensemble
    ensemble = EnsembleClassifier(
        custom_model=custom_model,
        custom_vocab=vocab,
        custom_id2label=id2label,
        pretrained_models=pretrained_models,
        device=device
    )
    
    print(f"\nEnsemble ready with {len(ensemble.model_names)} models:")
    for name in ensemble.model_names:
        print(f"  - {name}")
    
    # Test predictions
    test_texts = [
        "sit down please",
        "go left",
        "stand up",
        "turn around",
    ]
    
    print("\n" + "-"*70)
    print("Making predictions...\n")
    
    predictions, confidences, details = ensemble.predict_with_probabilities(test_texts)
    
    for text, pred, conf in zip(test_texts, predictions, confidences):
        print(f"'{text:25s}' → {pred:15s} (confidence: {conf:.0%})")
    
    # Show voting breakdown for first sample
    print("\n" + "-"*70)
    print("Detailed voting breakdown for first sample:\n")
    voting_detail = details['voting_details'][0]
    print(f"Text: {voting_detail['text']}")
    print(f"Predicted: {voting_detail['final_prediction_label']}")
    print(f"\nPer-model votes:")
    for model_name in sorted(voting_detail['votes'].keys()):
        vote_id = voting_detail['votes'][model_name]
        vote_label = id2label[vote_id]
        print(f"  {model_name:15s} → {vote_label}")
    print(f"\nVote counts:")
    for label, count in sorted(voting_detail['vote_counts'].items(), key=lambda x: -x[1]):
        print(f"  {label:15s}: {count} vote(s)")


def example_batch_analysis():
    """Batch prediction and voting analysis example."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Batch Predictions & Voting Analysis")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models (same as before)
    custom_model_path = Path('models/test_run/best.pth')
    ensemble_dir = Path('models/ensemble')
    
    if not custom_model_path.exists():
        print(f"ERROR: Custom model not found")
        return
    
    custom_model, vocab, id2label = load_custom_model(custom_model_path, device)
    pretrained_models = load_pretrained_models(ensemble_dir, device, len(id2label))
    
    ensemble = EnsembleClassifier(
        custom_model=custom_model,
        custom_vocab=vocab,
        custom_id2label=id2label,
        pretrained_models=pretrained_models,
        device=device
    )
    
    # Batch prediction
    test_texts = [
        "sit down please",
        "go left",
        "go right",
        "stand up",
        "turn around",
        "do a trick",
        "go forward",
        "go back",
    ]
    
    print(f"\nMaking predictions on {len(test_texts)} samples...\n")
    predictions, confidences, details = ensemble.predict_with_probabilities(test_texts)
    
    # Print results
    print("Predictions:")
    print("-" * 70)
    for text, pred, conf in zip(test_texts, predictions, confidences):
        status = "✓ CERTAIN" if conf == 1.0 else "~ UNCERTAIN" if conf < 0.75 else "✓ CONFIDENT"
        print(f"  '{text:25s}' → {pred:15s} [{conf:.0%}] {status}")
    
    # Analyze voting patterns
    analysis = analyze_voting_patterns(details['voting_details'])
    print_analysis(analysis)
    
    # Show disagreements
    disagreements = get_model_disagreements(details['voting_details'])
    print_disagreements(disagreements)
    
    # Compute diversity
    diversity = compute_ensemble_diversity_score(details['voting_details'])
    print(f"Ensemble diversity score: {diversity:.2f} (higher = more diverse predictions)")


def example_model_management():
    """Example of dynamically managing ensemble models."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Dynamic Model Management")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load base models
    custom_model_path = Path('models/test_run/best.pth')
    ensemble_dir = Path('models/ensemble')
    
    if not custom_model_path.exists():
        print(f"ERROR: Custom model not found")
        return
    
    custom_model, vocab, id2label = load_custom_model(custom_model_path, device)
    pretrained_models = load_pretrained_models(ensemble_dir, device, len(id2label))
    
    ensemble = EnsembleClassifier(
        custom_model=custom_model,
        custom_vocab=vocab,
        custom_id2label=id2label,
        pretrained_models=pretrained_models,
        device=device
    )
    
    print(f"\nInitial ensemble models: {ensemble.model_names}")
    
    # Remove a model
    if 'electra' in ensemble.model_names:
        print("\nRemoving 'electra' from ensemble...")
        ensemble.remove_pretrained_model('electra')
        print(f"Updated ensemble models: {ensemble.model_names}")
    
    # Test with reduced ensemble
    test_text = "sit down"
    predictions, confidences, details = ensemble.predict_with_probabilities([test_text])
    print(f"\nPrediction with reduced ensemble:")
    print(f"  '{test_text}' → {predictions[0]} ({confidences[0]:.0%})")
    
    # Add another model if available
    print(f"\nAvailable pretrained model types: {list(PRETRAINED_MODELS.keys())}")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("ENSEMBLE LEARNING EXAMPLES")
    print("="*70)
    print("\nThese examples demonstrate:")
    print("1. Basic ensemble prediction with majority voting")
    print("2. Batch predictions and voting analysis")
    print("3. Dynamic model management")
    print("\nNote: Make sure you've trained your models first!")
    print("  Custom model: python -m text_classification.train ...")
    print("  Ensemble models: python -m text_classification.train_ensemble ...")
    
    try:
        example_basic_prediction()
        # Uncomment to run other examples:
        # example_batch_analysis()
        # example_model_management()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nPlease train your models first. See ENSEMBLE_README.md for instructions.")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
