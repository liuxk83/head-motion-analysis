import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter1d

def load_ppmi_data(data_path = '../data/framewise_displacement_data.npz'):
    data = np.load(data_path, allow_pickle=True)
    
    # Convert the loaded data to a dictionary
    data_dict = {key: data[key].item() for key in data}

    pd_keys, pd_removed = filter_valid_subjects_ppmi(data_dict, 'PD')
    control_keys, control_removed = filter_valid_subjects_ppmi(data_dict, 'Control')

    print(f"Loaded {len(pd_keys)}/{(len(pd_keys) + len(control_keys))} PD subjects and {len(control_keys)}/{(len(control_keys) + len(control_keys))} Control subjects")

    return data_dict, pd_keys, control_keys

def load_hcp_data(data_path = '../data/hcp_aging_movement_data.npz', metadata_path = '../data/hcp_aging_movement_metadata.npz', metadata = False):
    data = np.load(data_path, allow_pickle=True)
    data_dict = {key: data[key].item() for key in data}
    
    print(f"Loaded {len(data_dict)} subject-run combinations")
    
    # Load metadata for reference
    if metadata:
        metadata = np.load(metadata_path, allow_pickle=True)
        print("Available data fields:", metadata['data_fields'])
        print("Available summary stats:", metadata['summary_stats'])
        return data_dict, metadata
    
    return data_dict

def load_hcp_metadata(data_path = '../data/hcp_aging_movement_metadata.npz'):
    data = np.load(data_path, allow_pickle=True)
    data_dict = {key: data[key].item() for key in data}
    return data_dict

# Filter and print removal counts for PD and Control groups
def filter_valid_subjects_ppmi(data_dict, group_name):
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

    return valid_keys, removed_count

# Plot examples
def plot_movement_metrics(movement_data, num_examples=5):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics = ['absolute_rms', 'relative_rms', 'framewise_displacement_equivalent']
    titles = ['Absolute RMS', 'Relative RMS', 'Framewise Displacement Equivalent']
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
    
    run1_keys = [key for key in movement_data.keys() if 'run-01' in key]
    run2_keys = [key for key in movement_data.keys() if 'run-02' in key]
    
    for i, metric in enumerate(metrics):
        # Plot for Run 1
        plot_count = 0
        for key in run1_keys[:num_examples]:
            if metric in movement_data[key] and len(movement_data[key][metric]) > 1:
                timeseries = movement_data[key][metric]
                if len(timeseries) > 10:  # Ensure reasonable length
                    # Z-score normalization
                    z = (timeseries - np.mean(timeseries)) / np.std(timeseries)
                    # Smooth the data
                    smoothed = gaussian_filter1d(z, sigma=1.0)
                    subject_id = key.split('_')[0]
                    axes[0, i].plot(smoothed, label=subject_id, 
                                   color=colors[plot_count % len(colors)], alpha=0.7)
                    plot_count += 1
                    
        axes[0, i].set_title(f'{titles[i]} - REST1', fontsize=12, fontweight='bold')
        axes[0, i].set_xlabel('Timepoint')
        axes[0, i].set_ylabel('Z-scored ' + titles[i])
        axes[0, i].set_xlim(0, 400)  # HCP typically has ~478 timepoints
        axes[0, i].grid(True, alpha=0.3)
        
        # Plot for Run 2
        plot_count = 0
        for key in run2_keys[:num_examples]:
            if metric in movement_data[key] and len(movement_data[key][metric]) > 1:
                timeseries = movement_data[key][metric]
                if len(timeseries) > 10:  # Ensure reasonable length
                    # Z-score normalization
                    z = (timeseries - np.mean(timeseries)) / np.std(timeseries)
                    # Smooth the data
                    smoothed = gaussian_filter1d(z, sigma=1.0)
                    subject_id = key.split('_')[0]
                    axes[1, i].plot(smoothed, label=subject_id, 
                                   color=colors[plot_count % len(colors)], alpha=0.7)
                    plot_count += 1
                    
        axes[1, i].set_title(f'{titles[i]} - REST2', fontsize=12, fontweight='bold')
        axes[1, i].set_xlabel('Timepoint')
        axes[1, i].set_ylabel('Z-scored ' + titles[i])
        axes[1, i].set_xlim(0, 400)
        axes[1, i].grid(True, alpha=0.3)
    
    # Add legends
    for ax in axes.flatten():
        ax.legend(loc='upper right', fontsize='small')
    
    plt.suptitle('HCP-Aging Movement Metrics: REST1 vs REST2', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Plot distribution of mean values
def plot_summary_statistics(movement_data):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Collect mean values for each run
    run1_abs_means = []
    run1_rel_means = []
    run2_abs_means = []
    run2_rel_means = []
    
    for key, data_dict in movement_data.items():
        metadata = data_dict.get('metadata', {})
        if 'run-01' in key:
            run1_abs_means.append(metadata.get('mean_absolute_rms', np.nan))
            run1_rel_means.append(metadata.get('mean_relative_rms', np.nan))
        elif 'run-02' in key:
            run2_abs_means.append(metadata.get('mean_absolute_rms', np.nan))
            run2_rel_means.append(metadata.get('mean_relative_rms', np.nan))
    
    # Remove NaN values
    run1_abs_means = [x for x in run1_abs_means if not np.isnan(x)]
    run1_rel_means = [x for x in run1_rel_means if not np.isnan(x)]
    run2_abs_means = [x for x in run2_abs_means if not np.isnan(x)]
    run2_rel_means = [x for x in run2_rel_means if not np.isnan(x)]
    
    # Plot absolute RMS distributions
    if run1_abs_means and run2_abs_means:
        axes[0].hist(run1_abs_means, bins=30, alpha=0.7, label='REST1', color='blue')
        axes[0].hist(run2_abs_means, bins=30, alpha=0.7, label='REST2', color='red')
        axes[0].set_xlabel('Mean Absolute RMS')
        axes[0].set_ylabel('Number of Subjects')
        axes[0].set_title('Distribution of Mean Absolute RMS')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Plot relative RMS distributions
    if run1_rel_means and run2_rel_means:
        axes[1].hist(run1_rel_means, bins=30, alpha=0.7, label='REST1', color='blue')
        axes[1].hist(run2_rel_means, bins=30, alpha=0.7, label='REST2', color='red')
        axes[1].set_xlabel('Mean Relative RMS')
        axes[1].set_ylabel('Number of Subjects')
        axes[1].set_title('Distribution of Mean Relative RMS')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    if run1_abs_means and run2_abs_means:
        print("\nSummary Statistics:")
        print(f"REST1 - Absolute RMS: Mean={np.mean(run1_abs_means):.4f}, Std={np.std(run1_abs_means):.4f}")
        print(f"REST2 - Absolute RMS: Mean={np.mean(run2_abs_means):.4f}, Std={np.std(run2_abs_means):.4f}")
        print(f"REST1 - Relative RMS: Mean={np.mean(run1_rel_means):.4f}, Std={np.std(run1_rel_means):.4f}")
        print(f"REST2 - Relative RMS: Mean={np.mean(run2_rel_means):.4f}, Std={np.std(run2_rel_means):.4f}")

# Create a scatter plot showing correlation between absolute and relative RMS
def plot_rms_correlation(movement_data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Collect data for scatter plots
    run1_abs, run1_rel = [], []
    run2_abs, run2_rel = [], []
    
    for key, data_dict in movement_data.items():
        metadata = data_dict.get('metadata', {})
        abs_mean = metadata.get('mean_absolute_rms', np.nan)
        rel_mean = metadata.get('mean_relative_rms', np.nan)
        
        if not (np.isnan(abs_mean) or np.isnan(rel_mean)):
            if 'run-01' in key:
                run1_abs.append(abs_mean)
                run1_rel.append(rel_mean)
            elif 'run-02' in key:
                run2_abs.append(abs_mean)
                run2_rel.append(rel_mean)
    
    # Plot correlations
    if run1_abs and run1_rel:
        axes[0].scatter(run1_abs, run1_rel, alpha=0.6, color='blue')
        axes[0].set_xlabel('Mean Absolute RMS')
        axes[0].set_ylabel('Mean Relative RMS')
        axes[0].set_title('REST1: Absolute vs Relative RMS')
        axes[0].grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr1 = np.corrcoef(run1_abs, run1_rel)[0, 1]
        axes[0].text(0.05, 0.95, f'r = {corr1:.3f}', transform=axes[0].transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    if run2_abs and run2_rel:
        axes[1].scatter(run2_abs, run2_rel, alpha=0.6, color='red')
        axes[1].set_xlabel('Mean Absolute RMS')
        axes[1].set_ylabel('Mean Relative RMS')
        axes[1].set_title('REST2: Absolute vs Relative RMS')
        axes[1].grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr2 = np.corrcoef(run2_abs, run2_rel)[0, 1]
        axes[1].text(0.05, 0.95, f'r = {corr2:.3f}', transform=axes[1].transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.show()

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
    """Dataset returning (T,) float32 tensor for a single metric + label, supporting multiple datasets with flexible metric mapping."""

    def __init__(self, datasets, metric_map, max_len=200):
        """
        datasets: list of (data_dict, keys, label) tuples
        metric_map: list of metric names (str), one per dataset
        max_len: int, pad/truncate all sequences to this length
        """
        if len(datasets) != len(metric_map):
            raise ValueError("datasets and metric_map must have the same length")
        self.max_len = max_len
        self.samples = []
        for (data_dict, keys, label), metric in zip(datasets, metric_map):
            for k in keys:
                rec = data_dict[k]
                if metric not in rec:
                    continue
                arr = np.array(rec[metric])
                # For PPMI, skip first value to match old behavior
                if arr.shape[0] > 1 and metric in ("framewise_displacement", "dvars", "rmsd"):
                    arr = arr[1:]
                # Z-score
                arr = (arr - arr.mean()) / (arr.std() + 1e-6)
                # Pad or truncate
                arr = arr[:max_len]
                if arr.shape[0] < max_len:
                    pad = np.zeros((max_len - arr.shape[0],), dtype=np.float32)
                    arr = np.concatenate([arr, pad])
                self.samples.append((arr.astype(np.float32), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)
