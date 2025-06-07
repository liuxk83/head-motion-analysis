"""
Train simple sequence models (GRU, LSTM, Transformer) to classify
Parkinson's disease (PD) vs Control from motion-derived rs-fMRI metrics.

Usage
-----
python pd_motion_classification.py --data framewise_displacement_data.npz
"""

import argparse
import random
import numpy as np
import pathlib
import torch
from sklearn.metrics import confusion_matrix, classification_report
import sys
sys.path.insert(1, 'src')

import data_processing as dp
import models_nn as mnn
import train_eval as te

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a simple sequence model on head motion data")
    parser.add_argument("--data", type=pathlib.Path, default=pathlib.Path("framewise_displacement_data.npz"),
                        help="Path to the input data file (default: framewise_displacement_data.npz)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training (default: 32)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs to train (default: 100)")
    parser.add_argument("--lr", type=float, default=2e-3,
                        help="Learning rate for the optimizer (default: 2e-3)")
    parser.add_argument("--max_len", type=int, default=100,
                        help="Maximum length of sequences (default: 100)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (default: cuda if available, else cpu)")
    parser.add_argument("--rng_seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    return parser.parse_args()

def main():

    args = parse_args()

    MAX_LEN = args.max_len  # truncate / zero-pad sequences to this length
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    DEVICE = args.device
    RNG_SEED = args.rng_seed

    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    torch.manual_seed(RNG_SEED)

    # Load the data from the .npz file
    data_file = 'framewise_displacement_data.npz'
    data = np.load(data_file, allow_pickle=True)

    # Convert the loaded data to a dictionary
    data_dict = {key: data[key].item() for key in data}

    pd_keys = dp.filter_valid_subjects(data_dict, 'PD')
    control_keys = dp.filter_valid_subjects(data_dict, 'Control')

    keys = pd_keys + control_keys
    random.shuffle(keys)

    dataset = dp.MotionDataset(data_dict, keys)
    train_loader, val_loader = te.train_val_split(
            dataset, val_size=0.2, random_state=RNG_SEED, batch_size=BATCH_SIZE
    )

    models = {
        "GRU": mnn.RNNClassifier(cell="gru"),
        "LSTM": mnn.RNNClassifier(cell="lstm"),
        "Transformer": mnn.TransformerClassifier()
    }

    for name, model in models.items():
        print(f"\nTraining {name}…")
        te.train(
            model, train_loader, val_loader,
            device=DEVICE, epochs=EPOCHS, lr=LR,
        )
        acc, bal_acc, auc, _, _ = te.evaluate(model, val_loader, device=DEVICE)
        print(
            f"{name} final  ACC={acc:.3f}  BAL_ACC={bal_acc:.3f}  AUC={auc:.3f}"
        )

        preds, targets = te.get_predictions(model, val_loader, device=DEVICE)
        cm = confusion_matrix(targets, preds)
        print("Confusion matrix:\n", cm)
        print(classification_report(targets, preds, digits=3))

if __name__ == "__main__":
    main()
