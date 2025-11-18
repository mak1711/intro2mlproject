"""
Ensemble prediction script using majority voting from multiple models.

This script:
1. Loads your custom trained model
2. Loads fine-tuned pretrained models
3. Uses majority voting to get final predictions
4. Shows voting breakdown and confidence scores
"""

import argparse
import json
from pathlib import Path
import torch
import sys

# Ensure project root is on path
proj_root = str(Path(__file__).resolve().parent.parent)
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from text_classification.model import TransformerClassifier
from text_classification.dataset import Vocab, TextCommandsDataset
from text_classification.utils import simple_tokenize
from text_classification.pretrained_models import get_pretrained_model, PRETRAINED_MODELS
from text_classification.ensemble import EnsembleClassifier


def load_custom_model(model_path: Path, device: torch.device):
    """Load custom transformer model."""
    cp = torch.load(model_path, map_location=device)
    
    label_map = cp.get('label_map')
    if label_map is None:
        raise RuntimeError('Checkpoint does not contain `label_map`')
    
    id2label = {int(v): k for k, v in label_map.items()}
    num_classes = len(id2label)
    
    # Load vocab
    itos = cp.get('vocab')
    if itos is None:
        raise RuntimeError('Checkpoint does not contain `vocab`')
    
    vocab = Vocab(tokens=[])
    vocab.itos = itos
    vocab.stoi = {t: i for i, t in enumerate(itos)}
    
    # Inspect checkpoint to infer model hyperparameters (d_model, num_layers, dim_feedforward, max_len)
    max_len = 128
    d_model = None
    num_layers = None
    dim_feedforward = None
    state = cp.get('model_state', {})
    try:
        if 'pos.pe' in state:
            pe = state['pos.pe']
            if hasattr(pe, 'shape') and len(pe.shape) >= 2:
                max_len = int(pe.shape[1])
                # last dim may be d_model
                if len(pe.shape) >= 3:
                    d_model = int(pe.shape[2])

        # infer d_model from embed.weight if available
        if d_model is None and 'embed.weight' in state:
            d_model = int(state['embed.weight'].shape[1])

        # infer num_layers from transformer_encoder layer keys
        layer_idxs = []
        for k in state.keys():
            if k.startswith('transformer_encoder.layers.'):
                parts = k.split('.')
                if len(parts) > 2 and parts[2].isdigit():
                    layer_idxs.append(int(parts[2]))
        if layer_idxs:
            num_layers = max(layer_idxs) + 1

        # infer dim_feedforward from linear1 weight shape if present
        for k in state.keys():
            if k.endswith('.linear1.weight'):
                dim_feedforward = int(state[k].shape[0])
                break
    except Exception:
        pass

    # Fallback defaults
    if d_model is None:
        d_model = 128
    if num_layers is None:
        num_layers = 2
    if dim_feedforward is None:
        dim_feedforward = max(2 * d_model, 256)

    # Choose nhead as a divisor of d_model (prefer larger heads)
    if d_model % 8 == 0:
        nhead = 8
    elif d_model % 4 == 0:
        nhead = 4
    elif d_model % 2 == 0:
        nhead = 2
    else:
        nhead = 1

    # Create and load model (use inferred hyperparameters to match checkpoint)
    model = TransformerClassifier(
        vocab_size=len(vocab),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        num_classes=num_classes,
        pad_idx=0,
        max_len=max_len
    )
    # load state dict and allow strict=False in case keys differ slightly
    model.load_state_dict(cp['model_state'], strict=True)
    model.to(device)
    model.eval()
    
    return model, vocab, id2label


def load_pretrained_models(ensemble_dir: Path, device: torch.device, num_classes: int):
    """Load all pretrained models from ensemble directory."""
    pretrained_models = {}
    ensemble_dir = Path(ensemble_dir)
    
    for model_type in PRETRAINED_MODELS.keys():
        model_path = ensemble_dir / model_type / 'best.pth'
        
        if not model_path.exists():
            continue
        
        try:
            print(f"Loading {model_type}...", end=' ')
            model_wrapper = get_pretrained_model(model_type, num_classes, device)
            model_wrapper.load_state_dict(torch.load(model_path, map_location=device))
            model_wrapper.eval_mode()
            pretrained_models[model_type] = model_wrapper
            print("✓")
        except Exception as e:
            print(f"✗ (Error: {e})")
    
    return pretrained_models


def main():
    p = argparse.ArgumentParser(description='Ensemble prediction with majority voting')
    p.add_argument('--custom-model', '-m', default='models/test_run/best.pth',
                   help='Path to custom model checkpoint')
    p.add_argument('--ensemble-dir', '-e', default='models/ensemble',
                   help='Directory containing fine-tuned pretrained models')
    p.add_argument('--custom-max-len', type=int, default=32,
                   help='Max sequence length for custom model')
    p.add_argument('--pretrained-max-len', type=int, default=128,
                   help='Max sequence length for pretrained models')
    p.add_argument('--device', default=None,
                   help='torch device (cpu or cuda). Default: auto-detect')
    p.add_argument('--verbose', '-v', action='store_true',
                   help='Show detailed voting breakdown')
    p.add_argument('text', nargs='*',
                   help='Optional text to predict (if omitted, enters interactive mode)')
    args = p.parse_args()

    # Setup
    device = torch.device(args.device) if args.device else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    custom_model_path = Path(args.custom_model)
    if not custom_model_path.exists():
        raise FileNotFoundError(f'Custom model not found: {custom_model_path}')

    # Load custom model
    print(f'Loading custom model from {custom_model_path}...')
    custom_model, vocab, id2label = load_custom_model(custom_model_path, device)
    print(f'  Classes: {id2label}')

    # Load pretrained models
    print(f'\nLoading pretrained models from {args.ensemble_dir}...')
    pretrained_models = load_pretrained_models(args.ensemble_dir, device, len(id2label))
    print(f'  Loaded {len(pretrained_models)} pretrained models')

    # Create ensemble
    ensemble = EnsembleClassifier(
        custom_model=custom_model,
        custom_vocab=vocab,
        custom_id2label=id2label,
        pretrained_models=pretrained_models,
        device=device
    )
    
    print(f'\nEnsemble ready with {len(ensemble.model_names)} total models:')
    for name in ensemble.model_names:
        print(f'  - {name}')

    def do_predict(text: str):
        """Run prediction and display results."""
        predictions, confidences, details = ensemble.predict_with_probabilities(
            [text],
            use_custom=True,
            use_pretrained=len(pretrained_models) > 0,
            custom_max_len=args.custom_max_len,
            pretrained_max_len=args.pretrained_max_len
        )
        
        prediction = predictions[0]
        confidence = confidences[0]
        voting_detail = details['voting_details'][0]
        
        print(f"\nInput: {text}")
        print(f"Prediction: {prediction} (confidence: {confidence:.1%})")
        
        if args.verbose:
            print(f"\nVoting breakdown:")
            votes = voting_detail['votes']
            for model_name in sorted(votes.keys()):
                vote_class_id = votes[model_name]
                vote_label = id2label[vote_class_id]
                print(f"  {model_name:20s} → {vote_label}")
            
            print(f"\nVote counts:")
            vote_counts = voting_detail['vote_counts']
            for label, count in sorted(vote_counts.items(), key=lambda x: -x[1]):
                print(f"  {label:20s}: {count} votes")
            
            print(f"Agreement: {voting_detail['agreement_count']}/{voting_detail['total_models']} models")

    # Interactive or batch mode
    if args.text:
        txt = ' '.join(args.text)
        do_predict(txt)
    else:
        print("\n" + "="*60)
        print("Interactive mode. Enter text to classify (blank line to exit)")
        print("="*60)
        try:
            while True:
                s = input('\nEnter text: ').strip()
                if not s:
                    break
                do_predict(s)
        except (KeyboardInterrupt, EOFError):
            print('\nExiting')


if __name__ == '__main__':
    main()
