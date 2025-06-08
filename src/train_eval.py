import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm

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

def plot_training_results(history, val_loader, model, device, class_names=None):
    """
    Plot training curves and confusion matrix.
    history: dict with keys 'loss', 'acc', 'bal_acc', 'auc'
    """
    epochs = len(history['loss'])
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Loss
    axes[0, 0].plot(range(1, epochs+1), history['loss'], label='Train Loss', color='tab:blue')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)

    # Accuracy
    axes[0, 1].plot(range(1, epochs+1), history['acc'], label='Val Acc', color='tab:green')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True)

    # Balanced Accuracy
    axes[0, 2].plot(range(1, epochs+1), history['bal_acc'], label='Val Balanced Acc', color='tab:orange')
    axes[0, 2].set_title('Balanced Accuracy')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Balanced Accuracy')
    axes[0, 2].grid(True)

    # AUC
    axes[1, 0].plot(range(1, epochs+1), history['auc'], label='Val AUC', color='tab:red')
    axes[1, 0].set_title('AUC')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].grid(True)

    # Confusion Matrix
    acc, bal_acc, auc, preds, targets = evaluate(model, val_loader, device)
    cm = confusion_matrix(targets, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=axes[1, 1], cmap='Blues', colorbar=False)
    axes[1, 1].set_title('Confusion Matrix')

    # Hide last subplot if not used
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

def train(model, train_loader, val_loader, device, epochs=50, lr=1e-3, plot=True, class_names=None):
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()
    history = {'loss': [], 'acc': [], 'bal_acc': [], 'auc': []}
    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {ep:02d}/{epochs}", leave=False)
        for seq, lbl in pbar:
            seq, lbl = seq.to(device), lbl.to(device)
            loss = crit(model(seq), lbl)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running_loss += loss.item() * seq.size(0)
            pbar.set_postfix({'loss': loss.item()})
        avg_loss = running_loss / len(train_loader.dataset)
        acc, bal_acc, auc, _, _ = evaluate(model, val_loader, device)
        history['loss'].append(avg_loss)
        history['acc'].append(acc)
        history['bal_acc'].append(bal_acc)
        history['auc'].append(auc)
        tqdm.write(
            f"Epoch {ep:02d}/{epochs}  loss={avg_loss:.4f}  val_acc={acc:.3f}  val_bal_acc={bal_acc:.3f}  val_auc={auc:.3f}"
        )
    if plot:
        plot_training_results(history, val_loader, model, device, class_names=class_names)
