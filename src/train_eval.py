import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

def train_val_split(dataset, val_size=0.2, random_state=None, batch_size=32):
    """Split dataset into training and validation sets."""
    val_len = int(val_size * len(dataset))
    train_len = len(dataset) - val_len
    generator = torch.Generator().manual_seed(random_state)
    train_set, val_set = random_split(
        dataset, [train_len, val_len], generator=generator
    )

    # class-balanced sampler for the train split
    train_labels = [dataset.samples[i][1] for i in train_set.indices]
    class_counts = np.bincount(train_labels)
    weights = 1.0 / class_counts
    sample_wts = [weights[lbl] for lbl in train_labels]
    sampler = WeightedRandomSampler(
        sample_wts, num_samples=len(sample_wts), replacement=True
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    return train_loader, val_loader

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, targets = [], []
    for seq, lbl in loader:
        seq = seq.to(device)
        logits = model(seq)
        probs.extend(torch.sigmoid(logits).cpu().numpy().ravel())
        targets.extend(lbl.numpy())
    preds = (np.array(probs) > 0.5).astype(int)
    acc = accuracy_score(targets, preds)
    bal_acc = balanced_accuracy_score(targets, preds)
    try:
        auc = roc_auc_score(targets, probs)
    except ValueError:
        auc = float("nan")
    return acc, bal_acc, auc, preds, targets


@torch.no_grad()
def get_predictions(model, loader, device):
    """Return hard preds and ground-truth labels for *loader*."""
    _, _, _, preds, targets = evaluate(model, loader, device)
    return preds, targets

def train(model, train_loader, val_loader, device, epochs=50, lr=1e-3):
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()
    for ep in range(1, epochs + 1):
        model.train()
        for seq, lbl in train_loader:
            seq, lbl = seq.to(device), lbl.to(device)
            loss = crit(model(seq), lbl)
            optim.zero_grad()
            loss.backward()
            optim.step()
        if ep % 5 == 0 or ep == epochs:
            acc, bal_acc, auc, _, _ = evaluate(model, val_loader, device)
            print(
                f"Epoch {ep:02d}/{epochs}  val_acc={acc:.3f}  val_bal_acc={bal_acc:.3f}  val_auc={auc:.3f}"
            )
