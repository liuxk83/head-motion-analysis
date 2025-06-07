import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

import torch
from torch.utils.data import Dataset

MAX_LEN = 100  # truncate / zero-pad sequences to this length

# Filter and print removal counts for PD and Control groups
def filter_valid_subjects(data_dict, group_name, print_info=True):
    valid_keys = []
    removed_count = 0

    for key, value in data_dict.items():
        if value['participant_info'].get('group') != group_name:
            continue
        if len(value['framewise_displacement']) < 100 or len(value['dvars']) < 2:
            removed_count += 1
            continue
        if np.std(value['dvars']) <= 1e-6:
            removed_count += 1
            continue
        valid_keys.append(key)

    if print_info:
        print(f"{group_name}: Removed {removed_count} out of {len([v for v in data_dict.values() if v['participant_info'].get('group') == group_name])}")
    
    return valid_keys

# Plot examples
def plot_metrics(data_dict, pd_keys, control_keys, num_examples=5):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    metrics = ['framewise_displacement', 'dvars', 'rmsd']
    titles = ['Framewise Displacement', 'DVARS', 'RMSD']
    
    for i, metric in enumerate(metrics):
        # Plot for PD patients
        for key in pd_keys[:num_examples]:
            z = data_dict[key][metric][1:]
            z = (z - np.mean(z)) / np.std(z)
            smoothed = gaussian_filter1d(z, sigma=0.5)
            axes[0, i].plot(smoothed, label=f'{key}')
            # axes[0, i].plot(framewise_displacement_data[key][metric], label=f'{key}')
        axes[0, i].set_title(f'{titles[i]} for PD Patients')
        axes[0, i].set_xlabel('Timepoint')
        axes[0, i].set_ylabel(titles[i])
        axes[0, i].set_xlim(0, 200)
        axes[0, i].grid(True)
        
        # Plot for Control patients
        for key in control_keys[:num_examples]:
            z = data_dict[key][metric][1:]
            z = (z - np.mean(z)) / np.std(z)
            smoothed = gaussian_filter1d(z, sigma=0.5)
            axes[1, i].plot(smoothed, label=f'{key}')
            # axes[1, i].plot(framewise_displacement_data[key][metric], label=f'{key}')
        axes[1, i].set_title(f'{titles[i]} for Control Patients')
        axes[1, i].set_xlabel('Timepoint')
        axes[1, i].set_ylabel(titles[i])
        axes[1, i].set_xlim(0, 200)
        axes[1, i].grid(True)
    
    # Add legends
    for ax in axes.flatten():
        ax.legend(loc='upper right', fontsize='small')

    # axes[0, 1].set_ylim(0, 150)
    # axes[1, 1].set_ylim(0, 150)
    
    plt.tight_layout()
    plt.show()

# Plot each subject in a separate subplot
def plot_individual_subplots(data_dict, pd_keys, control_keys, metric='dvars', num_examples=10):
    total = num_examples * 2
    cols = 5
    rows = (total + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = axes.flatten()

    all_keys = [(key, 'PD') for key in pd_keys[:num_examples]] + [(key, 'Control') for key in control_keys[:num_examples]]

    for i, (key, group) in enumerate(all_keys):
        ax = axes[i]
        ax.plot(data_dict[key][metric])
        ax.set_title(f'{group}: {key}', fontsize=10)
        ax.set_ylim(0, 100)
        ax.set_xlabel('Timepoint')
        ax.set_ylabel(metric.upper())
        ax.grid(True)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# Plot standardized DVARS
def plot_standardized_dvars(data_dict, pd_keys, control_keys, num_examples=5):
    total = num_examples * 2
    cols = 5
    rows = (total + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = axes.flatten()

    all_keys = [(key, 'PD') for key in pd_keys[:num_examples]] + [(key, 'Control') for key in control_keys[:num_examples]]

    for i, (key, group) in enumerate(all_keys):
        ax = axes[i]
        dvars = data_dict[key]['framewise_displacement'][1:]
        dvars_std = (dvars - np.mean(dvars)) / np.std(dvars)
        smoothed = gaussian_filter1d(dvars_std[1:], sigma=1.0)
        ax.plot(smoothed) # dvars_std
        
        ax.set_title(f'{group}: {key}', fontsize=10)
        ax.set_ylim(-5, 5)
        # ax.set_ylim(0, 100)
        ax.set_xlabel('Timepoint')
        ax.set_ylabel('Standardized DVARS')
        ax.grid(True)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


class MotionDataset(Dataset):
    """Dataset returning (T, 3) float32 tensor + binary label."""

    def __init__(self, data_dict, keys):
        self.samples = []
        for k in keys:
            rec = data_dict[k]
            fd, dv, rmsd = (
                np.array(rec[m][1:]) for m in ("framewise_displacement", "dvars", "rmsd")
            )

            def z(x):
                return (x - x.mean()) / (x.std() + 1e-6)

            seq = np.stack([z(fd[:MAX_LEN]), z(dv[:MAX_LEN]), z(rmsd[:MAX_LEN])], axis=1)
            # pad with zeros if shorter than MAX_LEN
            if seq.shape[0] < MAX_LEN:
                pad = np.zeros((MAX_LEN - seq.shape[0], 3), dtype=np.float32)
                seq = np.vstack([seq, pad])
            label = 1 if rec["participant_info"].get("group") == "PD" else 0
            self.samples.append((seq.astype(np.float32), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)
