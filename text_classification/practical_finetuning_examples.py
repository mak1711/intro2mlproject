"""
Practical Example: Fine-tuning & Using the Ensemble

This shows the complete workflow from fine-tuning to predictions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import torch
from text_classification.model import TransformerClassifier
from text_classification.dataset import Vocab, TextCommandsDataset
from text_classification.pretrained_models import get_pretrained_model
from text_classification.ensemble import EnsembleClassifier


def example_1_understand_finetuning():
    """
    Example 1: Understand what fine-tuning does
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Understanding Fine-tuning")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\nWhat is fine-tuning?")
    print("-" * 70)
    print("""
Fine-tuning = Taking a pre-trained model and training its LAST LAYERS
on YOUR specific dataset.

Why is this better than training from scratch?
1. Early layers already know language patterns (from training on huge corpus)
2. Only need to adapt last layers to your specific task
3. Much faster (weeks → hours)
4. Better accuracy (less data needed)
5. More stable (transfer learning)

Example Architecture (BERT):
  Input: "sit down"
           ↓
  [Embedding layer] ← FROZEN (keep pre-trained knowledge)
  [Layer 0] ← FROZEN
  [Layer 1] ← FROZEN
  [Layer 2] ← FROZEN
  [Layer 3] ← FROZEN
  [Layer 4] ← FROZEN
  [Layer 5] ← FROZEN
  [Layer 6] ← FROZEN
  [Layer 7] ← FROZEN
  [Layer 8] ← TRAIN THIS (adapt to your data)
  [Layer 9] ← TRAIN THIS
  [Layer 10] ← TRAIN THIS
  [Layer 11] ← TRAIN THIS
           ↓
  [Classification Head] ← TRAIN THIS (maps to your classes)
           ↓
  Output: "sit" (with confidence)
    """)
    
    print("\nWhat gets trained?")
    print("-" * 70)
    
    # Load a model to show the difference
    model = get_pretrained_model('bert', num_classes=9, device=device)
    
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"Total BERT parameters: {total_params:,}")
    print(f"Total BERT parameters: {total_params/1e6:.1f}M")
    
    # If we trained all
    print(f"\nIf we trained ALL layers:")
    print(f"  Training: 110,000,000 parameters")
    print(f"  Time: Very long (~24 hours on GPU)")
    print(f"  Memory: 2-3GB GPU")
    
    # If we fine-tune
    trainable = sum(p.numel() for p in model.model.parameters() 
                   if 'classifier' in str(type(p)))
    print(f"\nIf we FINE-TUNE (train last 4 layers + head):")
    print(f"  Training: ~15,000,000 parameters (13.6%)")
    print(f"  Time: ~3 hours on GPU for 3 epochs")
    print(f"  Memory: 1-1.5GB GPU")
    
    print("\n✨ Fine-tuning is MUCH more efficient! ✨")


def example_2_training_workflow():
    """
    Example 2: Show the training workflow
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Training Workflow")
    print("="*70)
    
    print("""
Step 1: Prepare Your Dataset
  - CSV file with 'text' and 'label' columns
  - Example: 
    text,label
    "sit down please",sit
    "stand up",stand
    
Step 2: Choose Model & Hyperparameters
  - Which models? (BERT, DistilBERT, RoBERTa, etc.)
  - How many epochs? (3-5 typical)
  - What learning rate? (2e-5 typical)
  - How many layers to train? (last 3-4 layers)
  
Step 3: Run Training Script
  python -m text_classification.train_ensemble \\
      --csv data/your_dataset.csv \\
      --models distilbert roberta \\
      --epochs 3 \\
      --unfreeze-last-n 4 \\
      --lr 2e-5
  
Step 4: Monitor Training
  - Logs show training loss/accuracy per epoch
  - Validation accuracy checked after each epoch
  - Best model automatically saved
  - Early stopping if no improvement
  
Step 5: Use Trained Models
  - Models saved to models/ensemble/
  - Use with predict_ensemble.py
  - Or load in Python code
  
EXAMPLE TRAINING OUTPUT:

Model: distilbert
Loading distilbert...

Fine-tuning Configuration:
  Total encoder layers: 6
  Layers to freeze: 2
  Layers to train: 4 (last N)
  Froze 2 encoder layers
  
  Trainable parameters: 30,754,050 / 66,955,778 (45.9%)

Epoch 1/3 - Training:
  [████████████] 50/50 batches
  loss=0.8234, acc=0.7845

Epoch 1/3 - Validation:
  Val Acc: 0.8234, Val Loss: 0.6432
  Saved best distilbert model (acc: 0.8234)

Epoch 2/3 - Training:
  [████████████] 50/50 batches
  loss=0.4532, acc=0.9123

Epoch 2/3 - Validation:
  Val Acc: 0.8645, Val Loss: 0.5234
  Saved best distilbert model (acc: 0.8645)

Epoch 3/3 - Training:
  [████████████] 50/50 batches
  loss=0.3124, acc=0.9456

Epoch 3/3 - Validation:
  Val Acc: 0.8567, Val Loss: 0.5789
  Early stopping for distilbert

✅ Training complete for distilbert!
Trained model saved to: models/ensemble/distilbert/best.pth
    """)


def example_3_hyperparameter_guide():
    """
    Example 3: Guide to choosing hyperparameters
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Hyperparameter Guide")
    print("="*70)
    
    print("""
KEY HYPERPARAMETERS FOR FINE-TUNING:

1. --unfreeze-last-n (how many layers to train)
   ────────────────────────────────────────────
   
   What it does: Controls how many layers are trainable
   
   Typical values:
   - 2-3: Very conservative, good for small datasets (<500 samples)
   - 4-5: Balanced, good for medium datasets (500-5000 samples)
   - 6-8: Aggressive, good for large datasets (>5000 samples)
   
   Effect:
   - Higher: Better accuracy but risk overfitting
   - Lower: More stable but may underfit
   
   My recommendation:
   - Start with 4
   - If overfitting: reduce to 2-3
   - If underfitting: increase to 6-8


2. --lr (learning rate)
   ──────────────────
   
   What it does: Controls how much weights change per update
   
   Typical values:
   - 1e-5 (0.00001): Conservative
   - 2e-5 (0.00002): Default, balanced
   - 5e-5 (0.00005): Aggressive
   
   Effect:
   - Too low: Very slow learning
   - Too high: May diverge or oscillate
   
   My recommendation:
   - Start with 2e-5
   - If overfitting: reduce to 1e-5
   - If underfitting: increase to 3e-5 or 5e-5
   
   ⚠️ IMPORTANT: Fine-tuning learning rates are 10-100x LOWER
                 than training from scratch!


3. --epochs (number of passes through data)
   ──────────────────────────────────────────
   
   What it does: How many times to train through entire dataset
   
   Typical values:
   - 1-2: Quick, but limited learning
   - 3-4: Balanced (recommended)
   - 5+: More learning but risk overfitting
   
   Effect:
   - More epochs: Better training accuracy
   - But: Risk overfitting after peak validation accuracy
   - Solution: Use early stopping (automatic)
   
   My recommendation:
   - Small dataset (<500): 4-5 epochs
   - Medium dataset (500-5000): 3 epochs
   - Large dataset (>5000): 2 epochs


4. --batch-size (samples per gradient update)
   ──────────────────────────────────────────
   
   What it does: How many samples to process before updating
   
   Typical values:
   - 8: Very memory efficient
   - 16: Good balance (default)
   - 32: More stable, more memory
   - 64: Very stable, high memory
   
   Effect:
   - Larger batch: More stable training
   - Smaller batch: Noisier but may escape local minima
   
   My recommendation:
   - Start with 16
   - If out of memory: reduce to 8
   - If stable but want better accuracy: increase to 32


5. --weight-decay (L2 regularization)
   ──────────────────────────────────
   
   What it does: Prevents overfitting by penalizing large weights
   
   Typical values:
   - 0.001: Light regularization
   - 0.01: Medium (default)
   - 0.1: Strong regularization
   
   Effect:
   - Higher: More regularization, simpler model
   - Lower: Less regularization, more complex model
   
   My recommendation:
   - 0.01 for most cases
   - 0.05+ if overfitting
   - 0.001 if underfitting


COMMON SCENARIOS & RECOMMENDED SETTINGS:

Scenario: Small dataset (<500 samples)
─────────────────────────────────────
--unfreeze-last-n 2-3
--lr 1e-5
--epochs 5
--batch-size 8
--weight-decay 0.05
Goal: Avoid overfitting

Scenario: Medium dataset (500-5000 samples)
────────────────────────────────────────────
--unfreeze-last-n 4
--lr 2e-5
--epochs 3
--batch-size 16
--weight-decay 0.01
Goal: Balance accuracy and stability

Scenario: Large dataset (>5000 samples)
───────────────────────────────────────
--unfreeze-last-n 6-8
--lr 3e-5
--epochs 2
--batch-size 32
--weight-decay 0.001
Goal: Maximize learning

Scenario: Limited GPU memory
─────────────────────────────
--unfreeze-last-n 2-3
--lr 2e-5
--epochs 3
--batch-size 4-8
--max-len 64
Goal: Fit in memory
    """)


def example_4_using_trained_models():
    """
    Example 4: How to use trained models for predictions
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Using Trained Models")
    print("="*70)
    
    print("""
After training with train_ensemble.py, your models are saved at:

models/
└── ensemble/
    ├── distilbert/
    │   └── best.pth         ← Trained DistilBERT weights
    ├── roberta/
    │   └── best.pth         ← Trained RoBERTa weights
    └── albert/
        └── best.pth         ← Trained ALBERT weights

To use them with predictions:

METHOD 1: Using Command Line
─────────────────────────────

python -m text_classification.predict_ensemble \\
    --custom-model models/my_custom/best.pth \\
    --ensemble-dir models/ensemble \\
    --verbose "sit down please"

Output:
  Input: sit down please
  Prediction: sit (confidence: 100%)
  
  Voting breakdown:
    custom               → sit
    distilbert           → sit
    roberta              → sit
  
  Vote counts:
    sit                  : 3 votes
  
  Agreement: 3/3 models


METHOD 2: Using Python API
───────────────────────────

from text_classification.ensemble import EnsembleClassifier
from text_classification.predict_ensemble import (
    load_custom_model,
    load_pretrained_models
)
from pathlib import Path
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load models
custom_model, vocab, id2label = load_custom_model(
    Path('models/my_custom/best.pth'), device
)
pretrained_models = load_pretrained_models(
    Path('models/ensemble'), device, len(id2label)
)

# Create ensemble
ensemble = EnsembleClassifier(
    custom_model, vocab, id2label, pretrained_models, device
)

# Make predictions
texts = ["sit down", "go left", "stand up"]
predictions, confidences, details = ensemble.predict_with_probabilities(texts)

for text, pred, conf in zip(texts, predictions, confidences):
    print(f"{text:20s} → {pred:15s} ({conf:.0%})")


METHOD 3: Batch Processing
───────────────────────────

# Process large dataset
from pathlib import Path
import torch
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load ensemble (as above)
ensemble = EnsembleClassifier(...)

# Load your data
texts = [...]  # list of texts to classify

# Predict
predictions, confidences, details = ensemble.predict_with_probabilities(
    texts,
    custom_max_len=32,
    pretrained_max_len=128
)

# Save results
results = []
for text, pred, conf in zip(texts, predictions, confidences):
    results.append({
        'text': text,
        'prediction': pred,
        'confidence': conf
    })

with open('predictions.json', 'w') as f:
    json.dump(results, f, indent=2)


INTERPRETING PREDICTIONS:

Confidence = proportion of models that voted for the prediction

Example 1: 100% confidence
  Input: "sit down"
  All models predicted: "sit"
  Confidence: 4/4 = 100%
  ✅ Very confident

Example 2: 75% confidence
  Input: "sit down"
  3 models predicted: "sit"
  1 model predicted: "stand"
  Confidence: 3/4 = 75%
  ⚠️ Reasonably confident (1 model disagrees)

Example 3: 50% confidence
  Input: "sit down"
  2 models predicted: "sit"
  2 models predicted: "stand"
  Confidence: 2/4 = 50%
  ⚠️ Uncertain (models strongly disagree)

What to do with low confidence?
- Option 1: Accept prediction but note low confidence
- Option 2: Ask for human review
- Option 3: Retrain with better data
    """)


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("FINE-TUNING & ENSEMBLE EXAMPLES")
    print("="*70)
    
    example_1_understand_finetuning()
    example_2_training_workflow()
    example_3_hyperparameter_guide()
    example_4_using_trained_models()
    
    print("\n" + "="*70)
    print("✅ EXAMPLES COMPLETE")
    print("="*70)
    print("""
QUICK NEXT STEPS:

1. Check your dataset:
   - Make sure you have a CSV with 'text' and 'label' columns
   
2. Run training (pick one based on your data size):
   
   For quick test (any size):
     python -m text_classification.train_ensemble \\
         --csv data/commands_dataset.csv \\
         --models distilbert \\
         --epochs 1
   
   For real training:
     python -m text_classification.train_ensemble \\
         --csv data/commands_dataset.csv \\
         --models distilbert roberta \\
         --epochs 3
   
3. Make predictions:
   python -m text_classification.predict_ensemble \\
       --custom-model models/my_custom/best.pth \\
       --ensemble-dir models/ensemble \\
       "your text here"

Good luck! 🚀
    """)


if __name__ == '__main__':
    main()
