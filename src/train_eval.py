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
    eval_loss = nn.BCEWithLogitsLoss()(torch.tensor(probs), torch.tensor(targets))
    acc = accuracy_score(targets, preds)
    bal_acc = balanced_accuracy_score(targets, preds)
    try:
        auc = roc_auc_score(targets, probs)
    except ValueError:
        auc = float("nan")
    return acc, bal_acc, auc, preds, targets, eval_loss

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
    acc, bal_acc, auc, preds, targets, eval_loss = evaluate(model, val_loader, device)
    cm = confusion_matrix(targets, preds)
    
    # Calculate sensitivity and specificity
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=axes[1, 1], cmap='Blues', colorbar=False)
    axes[1, 1].set_title('Confusion Matrix')
    
    # Add sensitivity and specificity text
    axes[1, 1].text(0.02, 0.98, f'Sensitivity: {sensitivity:.3f}', 
                    transform=axes[1, 1].transAxes, fontsize=10, 
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    axes[1, 1].text(0.02, 0.88, f'Specificity: {specificity:.3f}', 
                    transform=axes[1, 1].transAxes, fontsize=10, 
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    # Print final metrics
    print(f"\n=== FINAL VALIDATION METRICS ===")
    print(f"Accuracy: {acc:.3f}")
    print(f"Balanced Accuracy: {bal_acc:.3f}")
    print(f"AUC: {auc:.3f}")
    print(f"Sensitivity: {sensitivity:.3f}")
    print(f"Specificity: {specificity:.3f}")
    
    # Hide last subplot if not used
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()
    
    return {
        'confusion_matrix': cm,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'auc': auc,
        'loss': eval_loss
    }
    
class LabelSmoothingBCELoss(nn.Module):
    """
    Binary Cross Entropy with Label Smoothing to prevent overconfidence.
    """
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, logits, targets):
        # smooth the targets: 0 -> smoothing/2, 1 -> 1 - smoothing/2
        targets_smooth = targets * (1 - self.smoothing) + self.smoothing / 2
        return nn.functional.binary_cross_entropy_with_logits(logits, targets_smooth)

def train(model, train_loader, val_loader, device, epochs=50, lr=1e-3, plot=True, class_names=None, crit=nn.BCEWithLogitsLoss(), weight_decay=0.05):
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='max', factor=0.5, patience=20, verbose=True
    )
    history = {'loss': [], 'acc': [], 'bal_acc': [], 'auc': []}
    
    # Detect environment and set up progress bars accordingly
    try:
        from IPython import get_ipython, display
        if get_ipython() is not None and get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
            # Jupyter notebook - use notebook-specific handling
            from tqdm.notebook import tqdm as notebook_tqdm
            use_jupyter = True
        else:
            use_jupyter = False
    except ImportError:
        use_jupyter = False
    
    if use_jupyter:
        # Jupyter notebook version with minimal output
        from tqdm.notebook import tqdm as progress_bar
        
        # Main epoch progress bar
        epoch_pbar = progress_bar(range(1, epochs + 1), desc="Training Progress")
        
        for ep in epoch_pbar:
            model.train()
            running_loss = 0.0
            
            # Simple batch loop without inner progress bar to reduce output
            batch_losses = []
            for seq, lbl in train_loader:
                seq, lbl = seq.to(device), lbl.to(device)
                loss = crit(model(seq), lbl)
                optim.zero_grad()
                loss.backward()
                optim.step()
                running_loss += loss.item() * seq.size(0)
                batch_losses.append(loss.item())
            
            avg_loss = running_loss / len(train_loader.dataset)
            acc, bal_acc, auc, _, _, eval_loss = evaluate(model, val_loader, device)
            history['loss'].append(avg_loss)
            history['acc'].append(acc)
            history['bal_acc'].append(bal_acc)
            history['auc'].append(auc)
            
            # Update epoch progress bar with comprehensive metrics
            epoch_pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'val_acc': f'{acc:.3f}',
                'bal_acc': f'{bal_acc:.3f}',
                'auc': f'{auc:.3f}',
                'batch_loss': f'{np.mean(batch_losses[-5:]):.4f}'  # Recent batch loss
            })
            
            # scheduler.step(bal_acc)  # Reduce LR when bal_acc plateaus
            scheduler.step(eval_loss)  # Reduce LR when eval_loss plateaus
    
    else:
        # Terminal/IDE version with dual progress bars
        epoch_pbar = tqdm(range(1, epochs + 1), desc="Training Progress", position=0)
        
        for ep in epoch_pbar:
            model.train()
            running_loss = 0.0
            
            # Inner progress bar for batches
            batch_pbar = tqdm(train_loader, desc=f"Epoch {ep:02d}", leave=False, position=1)
            
            for seq, lbl in batch_pbar:
                seq, lbl = seq.to(device), lbl.to(device)
                loss = crit(model(seq), lbl)
                optim.zero_grad()
                loss.backward()
                optim.step()
                running_loss += loss.item() * seq.size(0)
                batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            avg_loss = running_loss / len(train_loader.dataset)
            acc, bal_acc, auc, _, _, eval_loss = evaluate(model, val_loader, device)
            history['loss'].append(avg_loss)
            history['acc'].append(acc)
            history['bal_acc'].append(bal_acc)
            history['auc'].append(auc)
            
            # Update epoch progress bar with metrics
            epoch_pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'val_acc': f'{acc:.3f}',
                'bal_acc': f'{bal_acc:.3f}',
                'auc': f'{auc:.3f}'
            })
            
            # scheduler.step(bal_acc)  # Reduce LR when bal_acc plateaus
            scheduler.step(eval_loss)  # Reduce LR when eval_loss plateaus
        
    if plot:
        plot_training_results(history, val_loader, model, device, class_names=class_names)
    
    return history

def train_quiet(model, train_loader, val_loader, device, epochs=50, lr=1e-3, plot=True, class_names=None, crit=nn.BCEWithLogitsLoss(), print_every=10):
    """
    Quiet training function for Jupyter notebooks - minimal output to avoid cell limit issues.
    Only prints progress every 'print_every' epochs.
    """
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='min', factor=0.8, patience=200)
    history = {'loss': [], 'acc': [], 'bal_acc': [], 'auc': []}
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Progress updates every {print_every} epochs.")
    print("=" * 50)
    
    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        
        # Train without progress bars
        for seq, lbl in train_loader:
            seq, lbl = seq.to(device), lbl.to(device)
            loss = crit(model(seq), lbl)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running_loss += loss.item() * seq.size(0)
        
        avg_loss = running_loss / len(train_loader.dataset)
        acc, bal_acc, auc, _, _, eval_loss = evaluate(model, val_loader, device)
        history['loss'].append(avg_loss)
        history['acc'].append(acc)
        history['bal_acc'].append(bal_acc)
        history['auc'].append(auc)
        
        # Print progress periodically
        if ep % print_every == 0 or ep == epochs:
            print(f"Epoch {ep:3d}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {acc:.3f} | Bal Acc: {bal_acc:.3f} | AUC: {auc:.3f} | LR: {scheduler.get_last_lr()[0]:.6f}", end="\r")
        
        # scheduler.step(bal_acc)  # Reduce LR when bal_acc plateaus
        scheduler.step(eval_loss)  # Reduce LR when eval_loss plateaus
    
    print()
    print("=" * 50)
    print("Training completed!")
    
    # Calculate final metrics including sensitivity and specificity
    acc, bal_acc, auc, preds, targets, eval_loss = evaluate(model, val_loader, device)
    cm = confusion_matrix(targets, preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\n=== FINAL VALIDATION METRICS ===")
    print(f"Accuracy: {acc:.3f}")
    print(f"Balanced Accuracy: {bal_acc:.3f}")
    print(f"AUC: {auc:.3f}")
    print(f"Sensitivity: {sensitivity:.3f}")
    print(f"Specificity: {specificity:.3f}")
    
    if plot:
        plot_training_results(history, val_loader, model, device, class_names=class_names)
    
    return history

@torch.no_grad()
def evaluate_by_source(model, dataset, device, batch_size=32):
    """
    Evaluate model performance broken down by dataset source.
    Returns dict with performance metrics for each source.
    """
    model.eval()
    results = {}
    
    # Handle Subset datasets from train_val_split
    if hasattr(dataset, 'dataset'):
        # This is a Subset, get the original dataset and indices
        original_dataset = dataset.dataset
        subset_indices = dataset.indices
        
        for source in ['ppmi', 'hcp']:
            source_indices = original_dataset.get_indices_by_source(source)
            # Get intersection of subset_indices and source_indices
            valid_indices = [i for i in subset_indices if i in source_indices]
            
            if not valid_indices:
                continue
                
            # Create subset loader with valid indices
            # Map back to subset space (indices within the subset)
            mapped_indices = [subset_indices.index(i) for i in valid_indices]
            subset = torch.utils.data.Subset(dataset, mapped_indices)
            loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
            
            # Evaluate on this subset
            acc, bal_acc, auc, preds, targets = evaluate(model, loader, device)
            
            # Count labels
            pd_count = sum(targets)
            control_count = len(targets) - pd_count
            
            results[source] = {
                'accuracy': acc,
                'balanced_accuracy': bal_acc,
                'auc': auc,
                'n_samples': len(targets),
                'n_pd': pd_count,
                'n_control': control_count,
                'predictions': preds,
                'targets': targets
            }
    else:
        # This is the original dataset
        for source in ['ppmi', 'hcp']:
            indices = dataset.get_indices_by_source(source)
            if not indices:
                continue
                
            # Create subset loader
            subset = torch.utils.data.Subset(dataset, indices)
            loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
            
            # Evaluate on this subset
            acc, bal_acc, auc, preds, targets = evaluate(model, loader, device)
            
            # Count labels
            pd_count = sum(targets)
            control_count = len(targets) - pd_count
            
            results[source] = {
                'accuracy': acc,
                'balanced_accuracy': bal_acc,
                'auc': auc,
                'n_samples': len(targets),
                'n_pd': pd_count,
                'n_control': control_count,
                'predictions': preds,
                'targets': targets
            }
    
    return results

@torch.no_grad()
def evaluate_val_by_source(model, val_loader, device):
    """
    Convenience function to evaluate validation set by source.
    """
    return evaluate_by_source(model, val_loader.dataset, device)

def plot_transformer_attention(model, val_loader, device, n_examples=4, class_names=None, save_data=True, save_path="attention_data.npz"):
    """
    Plot transformer attention weights for 1D motion sequences.
    
    Args:
        model: TransformerClassifierWithAttention model (must have get_attention_weights method)
        val_loader: validation data loader
        device: torch device
        n_examples: number of examples to show (max 4)
        class_names: list of class names (default: ['Control', 'PD'])
        save_data: whether to save plotting data for standalone use
        save_path: path to save the plotting data
    
    Usage:
        # After training:
        te.plot_transformer_attention(transformer_model, val_loader, device, 
                                    n_examples=4, class_names=["Controls", "PD"])
    """
    if not hasattr(model, 'get_attention_weights'):
        print("Model must be TransformerClassifierWithAttention with get_attention_weights method")
        return
    
    model.eval()
    
    # Get validation examples
    examples = []
    with torch.no_grad():
        for seq, lbl in val_loader:
            seq = seq.to(device)
            
            # Get predictions and attention weights
            logits, attention_weights = model.get_attention_weights(seq)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()
            
            # Store examples
            for i in range(seq.size(0)):
                examples.append({
                    'sequence': seq[i].cpu().numpy(),  # [seq_len, features]
                    'attention': attention_weights[-1][i].cpu().numpy(),  # Last layer attention [n_heads, seq_len, seq_len]
                    'true_label': lbl[i].item(),
                    'pred_label': preds[i].item(), 
                    'probability': probs[i].item()
                })
                
                if len(examples) >= n_examples:
                    break
            if len(examples) >= n_examples:
                break
    
    if len(examples) == 0:
        print("No examples found")
        return
    
    # Save data for standalone plotting if requested
    if save_data:
        save_attention_plotting_data(examples, class_names, save_path)
        print(f"💾 Attention plotting data saved to: {save_path}")
    
    # Generate the plots
    _plot_attention_examples(examples, class_names)

def save_attention_plotting_data(examples, class_names, save_path="attention_data.npz"):
    """
    Save all data needed for standalone attention plotting.
    """
    # Extract data from examples
    sequences = [ex['sequence'] for ex in examples]
    attentions = [ex['attention'] for ex in examples]
    true_labels = [ex['true_label'] for ex in examples]
    pred_labels = [ex['pred_label'] for ex in examples]
    probabilities = [ex['probability'] for ex in examples]
    
    # Save to compressed numpy format
    np.savez_compressed(
        save_path,
        sequences=sequences,
        attentions=attentions,
        true_labels=true_labels,
        pred_labels=pred_labels,
        probabilities=probabilities,
        class_names=class_names if class_names else ['Control', 'PD'],
        n_examples=len(examples)
    )

def _plot_attention_examples(examples, class_names):
    """
    Internal function to generate attention plots from examples data.
    """
    # Set up plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    if class_names is None:
        class_names = ['Control', 'PD']
    
    for idx, example in enumerate(examples[:4]):
        if idx >= len(axes):
            break
            
        seq = example['sequence']  # [seq_len, features]
        attention = example['attention']  # [n_heads, seq_len, seq_len]
        true_label = example['true_label']
        pred_label = example['pred_label']
        prob = example['probability']
        
        # Average attention across heads and compute attention received by each position
        print(f"Debug: attention shape = {attention.shape}")
        
        # Handle different possible attention tensor shapes
        if len(attention.shape) == 3:
            # Expected: [n_heads, seq_len, seq_len]
            avg_attention = attention.mean(axis=0)  # [seq_len, seq_len]
            attention_scores = avg_attention.mean(axis=0)  # [seq_len]
        elif len(attention.shape) == 2:
            # Case: [seq_len, seq_len] (single head or already averaged)
            attention_scores = attention.mean(axis=0)  # [seq_len]
        elif len(attention.shape) == 1:
            # Case: [seq_len] (already computed attention scores)
            attention_scores = attention
        else:
            print(f"Unexpected attention shape: {attention.shape}")
            # Fallback: create uniform attention
            attention_scores = np.ones(len(seq)) / len(seq)
        
        # Ensure attention_scores is 1D array
        attention_scores = np.asarray(attention_scores).flatten()
        
        # Normalize attention scores
        if len(attention_scores) > 1 and attention_scores.max() > attention_scores.min():
            attention_scores = (attention_scores - attention_scores.min()) / (attention_scores.max() - attention_scores.min())
        
        # Plot setup
        ax = axes[idx]
        time_points = np.arange(len(seq))
        
        # Handle 2D vs 1D input
        if seq.shape[1] == 2:
            # Two features: FD and RMSD
            fd_values = seq[:, 0]
            rmsd_values = seq[:, 1]
            
            # Create second y-axis
            ax2 = ax.twinx()
            
            # Plot FD (blue, left axis)
            ax.plot(time_points, fd_values, 'b-', linewidth=2, alpha=0.8, label='Framewise Displacement')
            ax.set_ylabel('Framewise Displacement', color='b', fontweight='bold')
            ax.tick_params(axis='y', labelcolor='b')
            
            # Plot RMSD (red, right axis)
            ax2.plot(time_points, rmsd_values, 'r-', linewidth=2, alpha=0.8, label='RMSD')
            ax2.set_ylabel('RMSD', color='r', fontweight='bold')
            ax2.tick_params(axis='y', labelcolor='r')
            
        else:
            # Single feature
            signal = seq.flatten()
            ax.plot(time_points, signal, 'b-', linewidth=2, alpha=0.8)
            ax.set_ylabel('Signal Value', fontweight='bold')
        
        # Add attention background coloring
        # Ensure we don't exceed sequence length
        max_time_points = min(len(attention_scores), len(seq))
        for t in range(max_time_points):
            alpha_val = attention_scores[t] * 0.4  # Scale for visibility
            ax.axvspan(t-0.5, t+0.5, alpha=alpha_val, color='purple', zorder=0)
        
        # Formatting
        ax.set_xlabel('Time Points', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Title with prediction info
        true_class = class_names[int(true_label)]
        pred_class = class_names[int(pred_label)]
        correct = "✓" if true_label == pred_label else "✗"
        
        ax.set_title(f'Example: {true_class} Patient', # {correct}
                    fontsize=12, fontweight='bold')
        
        # Color title based on correctness
        if true_label == pred_label:
            ax.title.set_color('green')
        else:
            ax.title.set_color('red')
    
    # Hide unused subplots
    for idx in range(len(examples), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('attention_visualization.png', dpi=300)
    plt.show()
    
    print("🔍 Transformer Attention Visualization")
    print("Purple background intensity = attention weight")
    print("Darker purple = model focuses more on those time points")