import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter1d
import random
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats

def load_ppmi_data(data_path = '../data/framewise_displacement_data.npz'):
    data = np.load(data_path, allow_pickle=True)
    
    # Convert the loaded data to a dictionary
    data_dict = {key: data[key].item() for key in data}

    pd_keys, pd_removed = filter_valid_subjects_ppmi(data_dict, 'PD')
    control_keys, control_removed = filter_valid_subjects_ppmi(data_dict, 'Control')

    print(f"Loaded {len(pd_keys)}/{(len(pd_keys) + len(control_keys))} PD subjects and {len(control_keys)}/{(len(control_keys) + len(control_keys))} Control subjects")

    return data_dict, pd_keys, control_keys

def load_hcp_data(data_path = '../data/hcp_aging_movement_data.npz', metadata_path = '../data/hcp_aging_movement_metadata.npz', metadata = False, run_1_only=True):
    data = np.load(data_path, allow_pickle=True)
    data_dict = {key: data[key].item() for key in data}
    
    # Filter to only run-01 if requested
    if run_1_only:
        data_dict = {key: value for key, value in data_dict.items() if 'run-01' in key}
        print(f"Loaded {len(data_dict)} run-01 subject combinations")
    else:
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

def detect_and_filter_spikes(signal, spike_threshold=3, method='interpolate', smooth_sigma=0):
    """
    Detect and filter spikes in motion signals.
    
    Args:
        signal: 1D array of motion data
        spike_threshold: Z-score threshold for spike detection (default: 3)
        method: 'interpolate', 'median', or 'clip'
        smooth_sigma: Optional Gaussian smoothing after spike removal (0 = no smoothing)
    
    Returns:
        Cleaned signal, spike mask (True where spikes were detected)
    """
    signal = np.array(signal)
    
    # Use robust statistics for spike detection
    median_val = np.median(signal)
    mad = np.median(np.abs(signal - median_val))  # Median Absolute Deviation
    
    # Convert MAD to approximate standard deviation (for normal distribution)
    robust_std = mad * 1.4826
    
    # Detect spikes using robust Z-score
    if robust_std > 0:
        z_scores = np.abs((signal - median_val) / robust_std)
        spike_mask = z_scores > spike_threshold
    else:
        spike_mask = np.zeros(len(signal), dtype=bool)
    
    # Filter spikes
    signal_clean = signal.copy()
    
    if np.any(spike_mask):
        spike_indices = np.where(spike_mask)[0]
        
        if method == 'interpolate':
            # Linear interpolation for isolated spikes
            for idx in spike_indices:
                if idx == 0:
                    # First timepoint - use next non-spike value
                    next_good = idx + 1
                    while next_good < len(signal) and spike_mask[next_good]:
                        next_good += 1
                    if next_good < len(signal):
                        signal_clean[idx] = signal[next_good]
                elif idx == len(signal) - 1:
                    # Last timepoint - use previous non-spike value
                    prev_good = idx - 1
                    while prev_good >= 0 and spike_mask[prev_good]:
                        prev_good -= 1
                    if prev_good >= 0:
                        signal_clean[idx] = signal[prev_good]
                else:
                    # Middle timepoint - interpolate between neighbors
                    signal_clean[idx] = (signal[idx-1] + signal[idx+1]) / 2
                    
        elif method == 'median':
            # Replace with local median
            for idx in spike_indices:
                window_start = max(0, idx - 2)
                window_end = min(len(signal), idx + 3)
                local_values = signal[window_start:window_end]
                # Exclude the spike itself
                local_values = np.concatenate([local_values[:idx-window_start], 
                                             local_values[idx-window_start+1:]])
                if len(local_values) > 0:
                    signal_clean[idx] = np.median(local_values)
                    
        elif method == 'clip':
            # Clip to threshold
            signal_clean[spike_mask] = np.clip(signal[spike_mask], 
                                             median_val - spike_threshold * robust_std,
                                             median_val + spike_threshold * robust_std)
    
    # Optional smoothing
    if smooth_sigma > 0:
        signal_clean = gaussian_filter1d(signal_clean, sigma=smooth_sigma)
    
    return signal_clean, spike_mask

def preprocess_motion_signal(signal, spike_threshold=3, spike_method='interpolate', 
                           smooth_sigma=0, dataset_type='ppmi'):
    """
    Complete preprocessing pipeline for motion signals.
    
    Args:
        signal: Raw motion signal
        spike_threshold: Z-score threshold for spike detection
        spike_method: Method for spike removal
        smooth_sigma: Gaussian smoothing parameter
        dataset_type: 'ppmi' or 'hcp' for dataset-specific processing
    
    Returns:
        Preprocessed signal
    """
    signal = np.array(signal)
    
    # Skip first value for PPMI framewise displacement
    if dataset_type == 'ppmi' and len(signal) > 1:
        signal = signal[1:]
    
    # Spike filtering
    signal_clean, spike_mask = detect_and_filter_spikes(
        signal, spike_threshold=spike_threshold, 
        method=spike_method, smooth_sigma=smooth_sigma
    )
    
    return signal_clean

class MotionDataset(Dataset):
    """Dataset returning (T, 2) float32 tensor for dual metrics + label, supporting PPMI and HCP datasets."""

    def __init__(self, datasets, max_len=200, spike_filtering=True, spike_threshold=3, 
                 spike_method='interpolate', smooth_sigma=0):
        """
        datasets: list of (data_dict, keys, label, dataset_type) tuples
            - dataset_type: 'ppmi' or 'hcp' to determine which metrics to use
        max_len: int, pad/truncate all sequences to this length
        spike_filtering: bool, whether to apply spike filtering
        spike_threshold: float or list of floats (one per dataset)
        spike_method: str or list of str (one per dataset)
        smooth_sigma: float or list of floats (one per dataset)
        
        PPMI uses: framewise_displacement + rmsd
        HCP uses: framewise_displacement_equivalent + relative_rms
        """
        self.max_len = max_len
        self.samples = []
        self.dataset_sources = []
        
        # Convert single values to lists if needed
        if not isinstance(spike_threshold, (list, tuple)):
            spike_threshold = [spike_threshold] * len(datasets)
        if not isinstance(spike_method, (list, tuple)):
            spike_method = [spike_method] * len(datasets)
        if not isinstance(smooth_sigma, (list, tuple)):
            smooth_sigma = [smooth_sigma] * len(datasets)
            
        # Validate parameter lengths
        if len(spike_threshold) != len(datasets):
            raise ValueError(f"spike_threshold length ({len(spike_threshold)}) must match datasets length ({len(datasets)})")
        if len(spike_method) != len(datasets):
            raise ValueError(f"spike_method length ({len(spike_method)}) must match datasets length ({len(datasets)})")
        if len(smooth_sigma) != len(datasets):
            raise ValueError(f"smooth_sigma length ({len(smooth_sigma)}) must match datasets length ({len(datasets)})")
        
        self.preprocessing_stats = {
            'total_spikes_detected': 0,
            'signals_with_spikes': 0,
            'total_signals': 0,
            'dataset_stats': {}
        }
        
        for dataset_idx, (data_dict, keys, label, dataset_type) in enumerate(datasets):
            # Get dataset-specific parameters
            ds_spike_threshold = spike_threshold[dataset_idx]
            ds_spike_method = spike_method[dataset_idx]
            ds_smooth_sigma = smooth_sigma[dataset_idx]
            
            # Initialize dataset-specific stats
            self.preprocessing_stats['dataset_stats'][dataset_type] = {
                'spikes_detected': 0,
                'signals_with_spikes': 0,
                'total_signals': 0,
                'spike_threshold': ds_spike_threshold,
                'spike_method': ds_spike_method,
                'smooth_sigma': ds_smooth_sigma
            }
            
            if dataset_type == 'ppmi':
                metrics = ['framewise_displacement', 'rmsd']
            elif dataset_type == 'hcp':
                metrics = ['framewise_displacement_equivalent', 'relative_rms']
            else:
                raise ValueError(f"Unknown dataset_type: {dataset_type}")
                
            for k in keys:
                rec = data_dict[k]
                # Check if both metrics exist
                if not all(metric in rec for metric in metrics):
                    continue
                    
                arrays = []
                for metric in metrics:
                    arr = np.array(rec[metric])
                    
                    # Apply preprocessing pipeline with dataset-specific parameters
                    if spike_filtering:
                        arr_clean = preprocess_motion_signal(
                            arr, spike_threshold=ds_spike_threshold,
                            spike_method=ds_spike_method, smooth_sigma=ds_smooth_sigma,
                            dataset_type=dataset_type
                        )
                        
                        # Track spike statistics
                        _, spike_mask = detect_and_filter_spikes(arr, ds_spike_threshold)
                        spikes_in_signal = np.sum(spike_mask)
                        self.preprocessing_stats['total_spikes_detected'] += spikes_in_signal
                        self.preprocessing_stats['dataset_stats'][dataset_type]['spikes_detected'] += spikes_in_signal
                        if np.any(spike_mask):
                            self.preprocessing_stats['signals_with_spikes'] += 1
                            self.preprocessing_stats['dataset_stats'][dataset_type]['signals_with_spikes'] += 1
                    else:
                        # Standard preprocessing without spike filtering
                        if dataset_type == 'ppmi' and arr.shape[0] > 1:
                            arr_clean = arr[1:]
                        else:
                            arr_clean = arr
                    
                    self.preprocessing_stats['total_signals'] += 1
                    self.preprocessing_stats['dataset_stats'][dataset_type]['total_signals'] += 1
                    
                    # Z-score normalization
                    arr_clean = (arr_clean - arr_clean.mean()) / (arr_clean.std() + 1e-6)
                    
                    # Pad or truncate
                    arr_clean = arr_clean[:max_len]
                    if arr_clean.shape[0] < max_len:
                        pad = np.zeros((max_len - arr_clean.shape[0],), dtype=np.float32)
                        arr_clean = np.concatenate([arr_clean, pad])
                    arrays.append(arr_clean)
                
                # Stack to create (T, 2) tensor
                combined = np.stack(arrays, axis=1).astype(np.float32)
                self.samples.append((combined, label))
                self.dataset_sources.append(dataset_type)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)
    
    def get_indices_by_source(self, source):
        """Get indices of samples from a specific dataset source."""
        return [i for i, src in enumerate(self.dataset_sources) if src == source]
    
    def print_preprocessing_stats(self):
        """Print statistics about spike filtering preprocessing."""
        stats = self.preprocessing_stats
        print(f"Preprocessing Statistics:")
        print(f"  Total signals processed: {stats['total_signals']}")
        print(f"  Signals with spikes: {stats['signals_with_spikes']} ({100*stats['signals_with_spikes']/stats['total_signals']:.1f}%)")
        print(f"  Total spikes detected: {stats['total_spikes_detected']}")
        print(f"  Average spikes per signal: {stats['total_spikes_detected']/stats['total_signals']:.2f}")
        
        print(f"\nDataset-specific Statistics:")
        for dataset_type, ds_stats in stats['dataset_stats'].items():
            print(f"  {dataset_type.upper()}:")
            print(f"    Signals processed: {ds_stats['total_signals']}")
            print(f"    Signals with spikes: {ds_stats['signals_with_spikes']} ({100*ds_stats['signals_with_spikes']/ds_stats['total_signals']:.1f}%)")
            print(f"    Spikes detected: {ds_stats['spikes_detected']}")
            print(f"    Parameters: threshold={ds_stats['spike_threshold']}, method='{ds_stats['spike_method']}', σ={ds_stats['smooth_sigma']}")

def create_ppmi_only_dataset(data_dict_ppmi, pd_keys, control_keys, max_len=200):
    """Create a dataset with only PPMI data for validation."""
    return MotionDataset(
        datasets=[
            (data_dict_ppmi, pd_keys, 1, 'ppmi'),           # PD
            (data_dict_ppmi, control_keys, 0, 'ppmi'),      # Control  
        ],
        max_len=max_len
    )

def create_single_feature_discrimination_dataset(data_dict_ppmi, control_keys, data_dict_hcp, 
                                               feature_idx=0, max_len=200, n_samples_per_dataset=None,
                                               spike_filtering=True, spike_threshold=[3, 3], 
                                               spike_method=['interpolate', 'interpolate'], 
                                               smooth_sigma=[0, 0]):
    """
    Create a dataset to test dataset discrimination using only one feature.
    
    Args:
        data_dict_ppmi: PPMI data dictionary
        control_keys: PPMI control subject keys
        data_dict_hcp: HCP data dictionary  
        feature_idx: which feature to use (0=framewise_displacement/equivalent, 1=rmsd/relative_rms)
        max_len: sequence length
        n_samples_per_dataset: number of samples to use from each dataset
        spike_filtering: whether to apply spike filtering
        spike_threshold: [ppmi_threshold, hcp_threshold]
        spike_method: [ppmi_method, hcp_method]
        smooth_sigma: [ppmi_sigma, hcp_sigma]
    
    Returns:
        Dataset with single feature, labels: 0=PPMI controls, 1=HCP samples
    """
    feature_names = {
        0: ("framewise_displacement", "framewise_displacement_equivalent"),
        1: ("rmsd", "relative_rms")
    }
    
    ppmi_metric, hcp_metric = feature_names[feature_idx]
    
    print(f"Creating single-feature discrimination test:")
    print(f"  Feature {feature_idx}: PPMI='{ppmi_metric}' vs HCP='{hcp_metric}'")
    
    # Get available sample counts
    n_ppmi_controls = len(control_keys)
    n_hcp_samples = len(list(data_dict_hcp.keys()))
    
    # Determine how many samples to use from each dataset
    if n_samples_per_dataset is None:
        n_samples_per_dataset = min(n_ppmi_controls, n_hcp_samples)
    else:
        n_samples_per_dataset = min(n_samples_per_dataset, n_ppmi_controls, n_hcp_samples)
    
    print(f"  Using {n_samples_per_dataset} samples from each dataset")
    if spike_filtering:
        print(f"  PPMI filtering: threshold={spike_threshold[0]}, method='{spike_method[0]}', σ={smooth_sigma[0]}")
        print(f"  HCP filtering:  threshold={spike_threshold[1]}, method='{spike_method[1]}', σ={smooth_sigma[1]}")
    
    # Randomly sample equal numbers from each dataset
    selected_ppmi = random.sample(control_keys, n_samples_per_dataset)
    selected_hcp = random.sample(list(data_dict_hcp.keys()), n_samples_per_dataset)
    
    # Create single-feature dataset
    class SingleFeatureDataset(Dataset):
        def __init__(self, datasets, feature_idx, max_len, spike_filtering, spike_threshold, spike_method, smooth_sigma):
            self.max_len = max_len
            self.samples = []
            
            for dataset_idx, (data_dict, keys, label, dataset_type) in enumerate(datasets):
                if dataset_type == 'ppmi':
                    metric = ["framewise_displacement", "rmsd"][feature_idx]
                elif dataset_type == 'hcp':
                    metric = ["framewise_displacement_equivalent", "relative_rms"][feature_idx]
                else:
                    raise ValueError(f"Unknown dataset_type: {dataset_type}")
                
                # Get dataset-specific filtering parameters
                ds_spike_threshold = spike_threshold[dataset_idx]
                ds_spike_method = spike_method[dataset_idx]
                ds_smooth_sigma = smooth_sigma[dataset_idx]
                
                for k in keys:
                    rec = data_dict[k]
                    if metric not in rec:
                        continue
                        
                    arr = np.array(rec[metric])
                    
                    # Apply preprocessing pipeline with dataset-specific parameters
                    if spike_filtering:
                        arr_clean = preprocess_motion_signal(
                            arr, spike_threshold=ds_spike_threshold,
                            spike_method=ds_spike_method, smooth_sigma=ds_smooth_sigma,
                            dataset_type=dataset_type
                        )
                    else:
                        # Standard preprocessing without spike filtering
                        if dataset_type == 'ppmi' and arr.shape[0] > 1:
                            arr_clean = arr[1:]
                        else:
                            arr_clean = arr
                    
                    # Z-score normalization
                    arr_clean = (arr_clean - arr_clean.mean()) / (arr_clean.std() + 1e-6)
                    
                    # Pad or truncate
                    arr_clean = arr_clean[:max_len]
                    if arr_clean.shape[0] < max_len:
                        pad = np.zeros((max_len - arr_clean.shape[0],), dtype=np.float32)
                        arr_clean = np.concatenate([arr_clean, pad])
                    
                    # Return as (T, 1) tensor for compatibility with models
                    arr_reshaped = arr_clean.reshape(-1, 1).astype(np.float32)
                    self.samples.append((arr_reshaped, label))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            x, y = self.samples[idx]
            return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)
    
    dataset = SingleFeatureDataset(
        datasets=[
            (data_dict_ppmi, selected_ppmi, 0, 'ppmi'),  # PPMI controls
            (data_dict_hcp, selected_hcp, 1, 'hcp'),     # HCP samples
        ],
        feature_idx=feature_idx,
        max_len=max_len,
        spike_filtering=spike_filtering,
        spike_threshold=spike_threshold,
        spike_method=spike_method,
        smooth_sigma=smooth_sigma
    )
    
    return dataset, selected_ppmi, selected_hcp, ppmi_metric, hcp_metric

def calculate_distribution_distance(signals1, signals2, metric='wasserstein'):
    """
    Calculate distance between two signal distributions.
    
    Args:
        signals1, signals2: Lists of signal arrays (not nested lists)
    """
    # Flatten signals to get distributions of values
    if isinstance(signals1[0], (list, np.ndarray)) and len(signals1[0].shape) >= 1:
        # signals1 is a list of signal arrays
        values1 = np.concatenate([sig.flatten() for sig in signals1])
    else:
        # signals1 is already flattened or is a nested list
        values1 = np.concatenate([np.array(sig).flatten() for sig in signals1])
    
    if isinstance(signals2[0], (list, np.ndarray)) and len(signals2[0].shape) >= 1:
        # signals2 is a list of signal arrays
        values2 = np.concatenate([sig.flatten() for sig in signals2])
    else:
        # signals2 is already flattened or is a nested list
        values2 = np.concatenate([np.array(sig).flatten() for sig in signals2])
    
    if metric == 'wasserstein':
        return stats.wasserstein_distance(values1, values2)
    elif metric == 'ks':
        return stats.ks_2samp(values1, values2).statistic
    elif metric == 'mean_diff':
        return abs(np.mean(values1) - np.mean(values2))
    elif metric == 'std_ratio':
        return abs(np.log(np.std(values1) / np.std(values2)))

def statistical_harmonization(data_dict_ppmi, control_keys, data_dict_hcp, 
                             hcp_keys=None, feature_idx=1, max_len=200, method='quantile_matching'):
    """
    Apply statistical harmonization to match PPMI and HCP distributions.
    
    Args:
        control_keys: PPMI control subject keys to use
        hcp_keys: HCP subject keys to use (if None, uses all available)
        method: 'quantile_matching', 'moment_matching', or 'z_score_alignment'
    """
    feature_info = {
        0: {'ppmi': 'framewise_displacement', 'hcp': 'framewise_displacement_equivalent'},
        1: {'ppmi': 'rmsd', 'hcp': 'relative_rms'}
    }
    
    ppmi_metric = feature_info[feature_idx]['ppmi']
    hcp_metric = feature_info[feature_idx]['hcp']
    
    # Extract raw signals
    ppmi_signals = []
    hcp_signals = []
    
    for key in control_keys:
        if ppmi_metric in data_dict_ppmi[key]:
            signal = np.array(data_dict_ppmi[key][ppmi_metric])
            if feature_idx == 0 and len(signal) > 1:  # Skip first for FD
                signal = signal[1:]
            ppmi_signals.append(signal[:max_len])
    
    # Use specified HCP keys or all available
    hcp_keys_to_use = hcp_keys if hcp_keys is not None else list(data_dict_hcp.keys())
    
    for key in hcp_keys_to_use:
        if key in data_dict_hcp and hcp_metric in data_dict_hcp[key]:
            signal = np.array(data_dict_hcp[key][hcp_metric])
            hcp_signals.append(signal[:max_len])
    
    if method == 'quantile_matching':
        # Match HCP quantiles to PPMI quantiles
        ppmi_flat = np.concatenate([sig for sig in ppmi_signals])
        hcp_flat = np.concatenate([sig for sig in hcp_signals])
        
        # Calculate quantiles
        quantiles = np.linspace(0, 1, 1000)
        ppmi_quantiles = np.quantile(ppmi_flat, quantiles)
        hcp_quantiles = np.quantile(hcp_flat, quantiles)
        
        # Transform HCP signals to match PPMI distribution
        hcp_harmonized = []
        for sig in hcp_signals:
            # Map each value to its quantile in HCP, then to corresponding PPMI value
            transformed = np.interp(sig, hcp_quantiles, ppmi_quantiles)
            hcp_harmonized.append(transformed)
        
        return ppmi_signals, hcp_harmonized, 'quantile_matching'
    
    elif method == 'moment_matching':
        # Match first two moments (mean and std)
        ppmi_flat = np.concatenate([sig for sig in ppmi_signals])
        hcp_flat = np.concatenate([sig for sig in hcp_signals])
        
        ppmi_mean, ppmi_std = np.mean(ppmi_flat), np.std(ppmi_flat)
        hcp_mean, hcp_std = np.mean(hcp_flat), np.std(hcp_flat)
        
        # Transform HCP: (x - hcp_mean) / hcp_std * ppmi_std + ppmi_mean
        hcp_harmonized = []
        for sig in hcp_signals:
            transformed = (sig - hcp_mean) / hcp_std * ppmi_std + ppmi_mean
            hcp_harmonized.append(transformed)
        
        return ppmi_signals, hcp_harmonized, 'moment_matching'
    
    elif method == 'z_score_alignment':
        # Z-score both, then rescale to match PPMI
        ppmi_harmonized = []
        hcp_harmonized = []
        
        # Z-score normalize each signal individually
        for sig in ppmi_signals:
            z_scored = (sig - sig.mean()) / (sig.std() + 1e-6)
            ppmi_harmonized.append(z_scored)
        
        for sig in hcp_signals:
            z_scored = (sig - sig.mean()) / (sig.std() + 1e-6)
            hcp_harmonized.append(z_scored)
        
        return ppmi_harmonized, hcp_harmonized, 'z_score_alignment'

def plot_harmonization_comparison(data_dict_ppmi, control_keys, data_dict_hcp,
                                feature_idx=1, max_len=200, n_examples=50,
                                best_params=None, harmonization_method=None):
    """
    Comprehensive visualization comparing original, filtered, and harmonized signals.
    """
    import matplotlib.pyplot as plt
    
    feature_info = {
        0: {'ppmi': 'framewise_displacement', 'hcp': 'framewise_displacement_equivalent', 'label': 'Framewise Displacement'},
        1: {'ppmi': 'rmsd', 'hcp': 'relative_rms', 'label': 'RMSD'}
    }
    
    info = feature_info[feature_idx]
    
    # 1. Original signals
    ppmi_orig, hcp_orig = [], []
    ppmi_sample = random.sample(control_keys, min(n_examples, len(control_keys)))
    hcp_sample = random.sample(list(data_dict_hcp.keys()), min(n_examples, len(data_dict_hcp.keys())))
    
    for key in ppmi_sample:
        if info['ppmi'] in data_dict_ppmi[key]:
            signal = np.array(data_dict_ppmi[key][info['ppmi']])
            if feature_idx == 0 and len(signal) > 1:
                signal = signal[1:]
            ppmi_orig.append(signal[:max_len])
    
    for key in hcp_sample:
        if info['hcp'] in data_dict_hcp[key]:
            signal = np.array(data_dict_hcp[key][info['hcp']])
            hcp_orig.append(signal[:max_len])
    
    # 2. Filtered signals (if best_params provided)
    ppmi_filtered, hcp_filtered = None, None
    if best_params:
        ppmi_filtered, hcp_filtered = [], []
        
        for key in ppmi_sample:
            if info['ppmi'] in data_dict_ppmi[key]:
                signal = np.array(data_dict_ppmi[key][info['ppmi']])
                signal_clean = preprocess_motion_signal(
                    signal, spike_threshold=best_params['spike_threshold'][0],
                    spike_method=best_params['spike_method'][0], 
                    smooth_sigma=best_params['smooth_sigma'][0],
                    dataset_type='ppmi'
                )[:max_len]
                ppmi_filtered.append(signal_clean)
        
        for key in hcp_sample:
            if info['hcp'] in data_dict_hcp[key]:
                signal = np.array(data_dict_hcp[key][info['hcp']])
                signal_clean = preprocess_motion_signal(
                    signal, spike_threshold=best_params['spike_threshold'][1],
                    spike_method=best_params['spike_method'][1], 
                    smooth_sigma=best_params['smooth_sigma'][1],
                    dataset_type='hcp'
                )[:max_len]
                hcp_filtered.append(signal_clean)
    
    # 3. Harmonized signals (if method provided)
    ppmi_harm, hcp_harm = None, None
    if harmonization_method:
        ppmi_harm, hcp_harm, method_name = statistical_harmonization(
            data_dict_ppmi, ppmi_sample, data_dict_hcp, 
            hcp_keys=hcp_sample, feature_idx=feature_idx, max_len=max_len, 
            method=harmonization_method
        )
    
    # Create comprehensive plot
    n_methods = 1 + (1 if best_params else 0) + (1 if harmonization_method else 0)
    fig, axes = plt.subplots(2, n_methods, figsize=(6*n_methods, 10))
    if n_methods == 1:
        axes = axes.reshape(-1, 1)
    
    methods = ['Original']
    data_sets = [(ppmi_orig, hcp_orig)]
    
    if best_params:
        methods.append('Optimized Filtering')
        data_sets.append((ppmi_filtered, hcp_filtered))
    
    if harmonization_method:
        methods.append(f'Statistical Harmonization ({harmonization_method})')
        data_sets.append((ppmi_harm, hcp_harm))
    
    for col, (method, (ppmi_data, hcp_data)) in enumerate(zip(methods, data_sets)):
        # Plot signals
        for sig in ppmi_data[:20]:
            axes[0, col].plot(sig, color='blue', alpha=0.3, linewidth=0.8)
        for sig in hcp_data[:20]:
            axes[0, col].plot(sig, color='red', alpha=0.3, linewidth=0.8)
        
        axes[0, col].set_title(f'{method}\nSignal Traces', fontweight='bold')
        axes[0, col].set_ylabel(info['label'])
        axes[0, col].grid(True, alpha=0.3)
        
        # Plot distributions
        ppmi_values = np.concatenate([sig for sig in ppmi_data])
        hcp_values = np.concatenate([sig for sig in hcp_data])
        
        axes[1, col].hist(ppmi_values, bins=50, alpha=0.6, color='blue', 
                         label=f'PPMI (μ={np.mean(ppmi_values):.3f}, σ={np.std(ppmi_values):.3f})', 
                         density=True)
        axes[1, col].hist(hcp_values, bins=50, alpha=0.6, color='red',
                         label=f'HCP (μ={np.mean(hcp_values):.3f}, σ={np.std(hcp_values):.3f})', 
                         density=True)
        
        # Calculate and show distance
        distance = calculate_distribution_distance(ppmi_data, hcp_data, 'wasserstein')
        axes[1, col].set_title(f'Value Distributions\nWasserstein Distance: {distance:.4f}', fontweight='bold')
        axes[1, col].set_xlabel(info['label'])
        axes[1, col].set_ylabel('Density')
        axes[1, col].legend()
        axes[1, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(f'{info["label"]}: Harmonization Comparison', fontsize=16, fontweight='bold', y=0.98)
    plt.show()
    
    # Print summary statistics
    print("Summary Statistics:")
    for method, (ppmi_data, hcp_data) in zip(methods, data_sets):
        ppmi_vals = np.concatenate([sig for sig in ppmi_data])
        hcp_vals = np.concatenate([sig for sig in hcp_data])
        
        mean_diff = abs(np.mean(ppmi_vals) - np.mean(hcp_vals))
        std_ratio = np.std(ppmi_vals) / np.std(hcp_vals)
        wasserstein_dist = calculate_distribution_distance(ppmi_data, hcp_data, 'wasserstein')
        
        print(f"\n{method}:")
        print(f"  Mean difference: {mean_diff:.4f}")
        print(f"  Std ratio (PPMI/HCP): {std_ratio:.3f}")
        print(f"  Wasserstein distance: {wasserstein_dist:.4f}")

def create_harmonized_dataset(data_dict_ppmi, control_keys, data_dict_hcp, 
                            feature_idx=1, max_len=200, harmonization_method='quantile_matching',
                            additional_filtering=None, balance_samples=True):
    """
    Create a dataset using statistical harmonization instead of/in addition to spike filtering.
    
    Args:
        balance_samples: If True, sample equal numbers from PPMI and HCP (default: True)
    """
    # Determine balanced sample sizes
    n_ppmi_controls = len(control_keys)
    n_hcp_available = len(list(data_dict_hcp.keys()))
    
    if balance_samples:
        n_samples_to_use = min(n_ppmi_controls, n_hcp_available)
        selected_ppmi_keys = random.sample(control_keys, n_samples_to_use)
        selected_hcp_keys = random.sample(list(data_dict_hcp.keys()), n_samples_to_use)
        
        print(f"Balanced sampling:")
        print(f"  PPMI controls available: {n_ppmi_controls}")
        print(f"  HCP samples available: {n_hcp_available}")
        print(f"  Using {n_samples_to_use} samples from each dataset")
    else:
        selected_ppmi_keys = control_keys
        selected_hcp_keys = list(data_dict_hcp.keys())
        print(f"Using all available samples:")
        print(f"  PPMI controls: {n_ppmi_controls}")
        print(f"  HCP samples: {n_hcp_available}")
    
    ppmi_signals, hcp_signals, method_name = statistical_harmonization(
        data_dict_ppmi, selected_ppmi_keys, data_dict_hcp,
        hcp_keys=selected_hcp_keys, feature_idx=feature_idx, max_len=max_len, method=harmonization_method
    )
    
    # Apply additional filtering if requested
    if additional_filtering:
        ppmi_processed = []
        for sig in ppmi_signals:
            processed = preprocess_motion_signal(
                sig, spike_threshold=additional_filtering.get('spike_threshold', 3),
                spike_method=additional_filtering.get('spike_method', 'interpolate'),
                smooth_sigma=additional_filtering.get('smooth_sigma', 0),
                dataset_type='ppmi'
            )
            ppmi_processed.append(processed)
        
        hcp_processed = []
        for sig in hcp_signals:
            processed = preprocess_motion_signal(
                sig, spike_threshold=additional_filtering.get('spike_threshold', 3),
                spike_method=additional_filtering.get('spike_method', 'interpolate'),
                smooth_sigma=additional_filtering.get('smooth_sigma', 0),
                dataset_type='hcp'
            )
            hcp_processed.append(processed)
        
        ppmi_signals, hcp_signals = ppmi_processed, hcp_processed
    
    # Create dataset
    class HarmonizedDataset(Dataset):
        def __init__(self, ppmi_signals, hcp_signals, max_len):
            self.samples = []
            
            # PPMI controls (label 0)
            for sig in ppmi_signals:
                # Z-score normalize
                sig_norm = (sig - sig.mean()) / (sig.std() + 1e-6)
                # Pad/truncate
                if len(sig_norm) < max_len:
                    pad = np.zeros(max_len - len(sig_norm))
                    sig_norm = np.concatenate([sig_norm, pad])
                else:
                    sig_norm = sig_norm[:max_len]
                
                sig_tensor = sig_norm.reshape(-1, 1).astype(np.float32)
                self.samples.append((sig_tensor, 0))
            
            # HCP samples (label 1)
            for sig in hcp_signals:
                # Z-score normalize
                sig_norm = (sig - sig.mean()) / (sig.std() + 1e-6)
                # Pad/truncate
                if len(sig_norm) < max_len:
                    pad = np.zeros(max_len - len(sig_norm))
                    sig_norm = np.concatenate([sig_norm, pad])
                else:
                    sig_norm = sig_norm[:max_len]
                
                sig_tensor = sig_norm.reshape(-1, 1).astype(np.float32)
                self.samples.append((sig_tensor, 1))
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            x, y = self.samples[idx]
            return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)
    
    dataset = HarmonizedDataset(ppmi_signals, hcp_signals, max_len)
    
    print(f"Created harmonized dataset using {method_name}")
    print(f"  PPMI samples: {len(ppmi_signals)}")
    print(f"  HCP samples: {len(hcp_signals)}")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Class balance: {len(ppmi_signals)}/{len(hcp_signals)} (PPMI/HCP)")
    if additional_filtering:
        print(f"  Additional filtering applied: {additional_filtering}")
    
    return dataset

def create_balanced_pd_classification_dataset(data_dict_ppmi, pd_keys, control_keys, data_dict_hcp,
                                             feature_idx=1, max_len=200, 
                                             harmonization_method='quantile_matching',
                                             balance_strategy='match_pd', 
                                             additional_filtering=None):
    """
    Create a balanced dataset for PD classification using harmonized HCP data as additional controls.
    
    Args:
        balance_strategy: 
            - 'match_pd': Total controls = number of PD patients
            - 'double_pd': Total controls = 2x number of PD patients  
            - 'equal_split': Equal numbers of PPMI controls, PD, and HCP
        
    Returns:
        Dataset with labels: 0=Controls (PPMI+HCP), 1=PD patients
    """
    n_pd = len(pd_keys)
    n_ppmi_controls = len(control_keys)
    n_hcp_available = len(list(data_dict_hcp.keys()))
    
    print(f"Creating balanced PD classification dataset:")
    print(f"  PPMI PD patients: {n_pd}")
    print(f"  PPMI controls: {n_ppmi_controls}")
    print(f"  HCP available: {n_hcp_available}")
    
    # Determine sampling strategy
    if balance_strategy == 'match_pd':
        # Total controls = number of PD patients
        n_total_controls_needed = n_pd
        n_hcp_to_use = max(0, n_total_controls_needed - n_ppmi_controls)
        n_ppmi_controls_to_use = min(n_ppmi_controls, n_total_controls_needed)
        
    elif balance_strategy == 'double_pd':
        # Total controls = 2x number of PD patients (less severe imbalance)
        n_total_controls_needed = 2 * n_pd
        n_hcp_to_use = max(0, n_total_controls_needed - n_ppmi_controls)
        n_ppmi_controls_to_use = min(n_ppmi_controls, n_total_controls_needed)
        
    elif balance_strategy == 'equal_split':
        # Equal numbers of PPMI controls, PD, and HCP
        n_per_group = min(n_pd, n_ppmi_controls, n_hcp_available)
        n_ppmi_controls_to_use = n_per_group
        n_hcp_to_use = n_per_group
        n_total_controls_needed = n_ppmi_controls_to_use + n_hcp_to_use
        
    else:
        raise ValueError(f"Unknown balance_strategy: {balance_strategy}")
    
    # Ensure we don't exceed available HCP samples
    n_hcp_to_use = min(n_hcp_to_use, n_hcp_available)
    
    print(f"  Strategy: {balance_strategy}")
    print(f"  PPMI controls to use: {n_ppmi_controls_to_use}")
    print(f"  HCP samples to use: {n_hcp_to_use}")
    print(f"  Total controls: {n_ppmi_controls_to_use + n_hcp_to_use}")
    print(f"  PD patients: {n_pd}")
    print(f"  Final ratio (Controls:PD): {(n_ppmi_controls_to_use + n_hcp_to_use)/n_pd:.1f}:1")
    
    # Sample the data
    selected_ppmi_controls = random.sample(control_keys, n_ppmi_controls_to_use) if n_ppmi_controls_to_use < n_ppmi_controls else control_keys
    selected_hcp_keys = random.sample(list(data_dict_hcp.keys()), n_hcp_to_use) if n_hcp_to_use > 0 else []
    selected_pd_keys = pd_keys  # Use all PD patients
    
    # Get PPMI control signals (will be used as reference for harmonization)
    ppmi_control_signals = []
    for key in selected_ppmi_controls:
        feature_info = {
            0: {'ppmi': 'framewise_displacement'},
            1: {'ppmi': 'rmsd'}
        }
        metric = feature_info[feature_idx]['ppmi']
        
        if metric in data_dict_ppmi[key]:
            signal = np.array(data_dict_ppmi[key][metric])
            if feature_idx == 0 and len(signal) > 1:  # Skip first for FD
                signal = signal[1:]
            ppmi_control_signals.append(signal[:max_len])
    
    # Get PD signals
    ppmi_pd_signals = []
    for key in selected_pd_keys:
        feature_info = {
            0: {'ppmi': 'framewise_displacement'},
            1: {'ppmi': 'rmsd'}
        }
        metric = feature_info[feature_idx]['ppmi']
        
        if metric in data_dict_ppmi[key]:
            signal = np.array(data_dict_ppmi[key][metric])
            if feature_idx == 0 and len(signal) > 1:  # Skip first for FD
                signal = signal[1:]
            ppmi_pd_signals.append(signal[:max_len])
    
    # Harmonize HCP to match PPMI distribution (using PPMI controls as reference)
    hcp_harmonized_signals = []
    if n_hcp_to_use > 0:
        print(f"  Harmonizing {n_hcp_to_use} HCP samples to match PPMI distribution...")
        
        # Get raw HCP signals
        feature_info = {
            0: {'hcp': 'framewise_displacement_equivalent'},
            1: {'hcp': 'relative_rms'}
        }
        hcp_metric = feature_info[feature_idx]['hcp']
        
        hcp_raw_signals = []
        for key in selected_hcp_keys:
            if hcp_metric in data_dict_hcp[key]:
                signal = np.array(data_dict_hcp[key][hcp_metric])
                hcp_raw_signals.append(signal[:max_len])
        
        # Apply harmonization (HCP -> PPMI distribution)
        if harmonization_method == 'quantile_matching':
            # Use PPMI controls as reference distribution
            ppmi_flat = np.concatenate([sig for sig in ppmi_control_signals])
            hcp_flat = np.concatenate([sig for sig in hcp_raw_signals])
            
            # Calculate quantiles
            quantiles = np.linspace(0, 1, 1000)
            ppmi_quantiles = np.quantile(ppmi_flat, quantiles)
            hcp_quantiles = np.quantile(hcp_flat, quantiles)
            
            # Transform HCP signals to match PPMI distribution
            for sig in hcp_raw_signals:
                transformed = np.interp(sig, hcp_quantiles, ppmi_quantiles)
                hcp_harmonized_signals.append(transformed)
        
        elif harmonization_method == 'moment_matching':
            # Match mean and std to PPMI controls
            ppmi_flat = np.concatenate([sig for sig in ppmi_control_signals])
            hcp_flat = np.concatenate([sig for sig in hcp_raw_signals])
            
            ppmi_mean, ppmi_std = np.mean(ppmi_flat), np.std(ppmi_flat)
            hcp_mean, hcp_std = np.mean(hcp_flat), np.std(hcp_flat)
            
            for sig in hcp_raw_signals:
                transformed = (sig - hcp_mean) / hcp_std * ppmi_std + ppmi_mean
                hcp_harmonized_signals.append(transformed)
    
    # Apply additional filtering if requested
    if additional_filtering:
        print(f"  Applying additional filtering...")
        
        # Filter PPMI controls
        ppmi_control_processed = []
        for sig in ppmi_control_signals:
            processed = preprocess_motion_signal(
                sig, spike_threshold=additional_filtering.get('spike_threshold', 3),
                spike_method=additional_filtering.get('spike_method', 'interpolate'),
                smooth_sigma=additional_filtering.get('smooth_sigma', 0),
                dataset_type='ppmi'
            )
            ppmi_control_processed.append(processed)
        ppmi_control_signals = ppmi_control_processed
        
        # Filter PD signals
        ppmi_pd_processed = []
        for sig in ppmi_pd_signals:
            processed = preprocess_motion_signal(
                sig, spike_threshold=additional_filtering.get('spike_threshold', 3),
                spike_method=additional_filtering.get('spike_method', 'interpolate'),
                smooth_sigma=additional_filtering.get('smooth_sigma', 0),
                dataset_type='ppmi'
            )
            ppmi_pd_processed.append(processed)
        ppmi_pd_signals = ppmi_pd_processed
        
        # Filter harmonized HCP signals
        hcp_harmonized_processed = []
        for sig in hcp_harmonized_signals:
            processed = preprocess_motion_signal(
                sig, spike_threshold=additional_filtering.get('spike_threshold', 3),
                spike_method=additional_filtering.get('spike_method', 'interpolate'),
                smooth_sigma=additional_filtering.get('smooth_sigma', 0),
                dataset_type='hcp'  # Use HCP filtering params even though it's harmonized
            )
            hcp_harmonized_processed.append(processed)
        hcp_harmonized_signals = hcp_harmonized_processed
    
    # Create dataset
    class BalancedPDDataset(Dataset):
        def __init__(self, ppmi_control_signals, hcp_harmonized_signals, ppmi_pd_signals, max_len):
            self.samples = []
            
            # Process all control signals (PPMI + harmonized HCP) -> Label 0
            for sig in ppmi_control_signals + hcp_harmonized_signals:
                # Z-score normalize
                sig_norm = (sig - sig.mean()) / (sig.std() + 1e-6)
                # Pad/truncate
                if len(sig_norm) < max_len:
                    pad = np.zeros(max_len - len(sig_norm))
                    sig_norm = np.concatenate([sig_norm, pad])
                else:
                    sig_norm = sig_norm[:max_len]
                
                sig_tensor = sig_norm.reshape(-1, 1).astype(np.float32)
                self.samples.append((sig_tensor, 0))  # Control label
            
            # Process PD signals -> Label 1
            for sig in ppmi_pd_signals:
                # Z-score normalize
                sig_norm = (sig - sig.mean()) / (sig.std() + 1e-6)
                # Pad/truncate
                if len(sig_norm) < max_len:
                    pad = np.zeros(max_len - len(sig_norm))
                    sig_norm = np.concatenate([sig_norm, pad])
                else:
                    sig_norm = sig_norm[:max_len]
                
                sig_tensor = sig_norm.reshape(-1, 1).astype(np.float32)
                self.samples.append((sig_tensor, 1))  # PD label
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            x, y = self.samples[idx]
            return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)
    
    dataset = BalancedPDDataset(ppmi_control_signals, hcp_harmonized_signals, ppmi_pd_signals, max_len)
    
    # Final statistics
    n_total_controls = len(ppmi_control_signals) + len(hcp_harmonized_signals)
    n_total_pd = len(ppmi_pd_signals)
    
    print(f"\nCreated balanced PD classification dataset:")
    print(f"  Total controls: {n_total_controls} (PPMI: {len(ppmi_control_signals)}, HCP harmonized: {len(hcp_harmonized_signals)})")
    print(f"  Total PD: {n_total_pd}")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Class balance ratio: {n_total_controls/n_total_pd:.2f}:1 (Controls:PD)")
    print(f"  Harmonization method: {harmonization_method}")
    
    dataset_info = {
        'n_ppmi_controls': len(ppmi_control_signals),
        'n_hcp_harmonized': len(hcp_harmonized_signals),
        'n_total_controls': n_total_controls,
        'n_pd': n_total_pd,
        'balance_ratio': n_total_controls/n_total_pd,
        'harmonization_method': harmonization_method,
        'balance_strategy': balance_strategy
    }
    
    return dataset, dataset_info
