import numpy as np
import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from Utils.plot_config import setup_plot_style, COLORS

# Set up standardized plot style
setup_plot_style()
    

# Precision/recall curve or ROC curve
# Localization error (absolute error, median error), how sharp are the detected change points
# Overlay plots of detected change points on the signal

def precision(detected_cps, true_cps, tol):
    """Calculate precision of detected change points."""
    true_positives = 0
    for cp in detected_cps:
        if any(abs(cp - true_cp) <= tol for true_cp in true_cps):
            true_positives += 1
    precision_value = true_positives / len(detected_cps) if detected_cps else 0
    return precision_value

def recall(detected_cps, true_cps, tol):
    """Calculate recall of detected change points."""
    true_positives = 0
    for true_cp in true_cps:
        if any(abs(cp - true_cp) <= tol for cp in detected_cps):
            true_positives += 1
    recall_value = true_positives / len(true_cps) if true_cps else 0
    return recall_value

def F1_score(precision_value, recall_value):
    """Calculate F1 score from precision and recall."""
    if precision_value + recall_value == 0:
        return 0
    return 2 * (precision_value * recall_value) / (precision_value + recall_value)

def read_change_points(file_path):
    """Read change points from a CSV file."""
    change_points = []
    with open(file_path, 'r') as f:
        for line in f:
            if "means" in line:
                break
            change_points.append(int(line.strip()))
    return change_points

def read_true_params(file_path):
    """Read true change point parameters from a CSV file with region parameters.
    
    Calculates change points from cumulative sum of region lengths.
    Works with both pretty_data_params.csv and realistic_data_params.csv formats.
    """
    
    # Read the CSV file with region parameters
    df = pd.read_csv(file_path)
    
    # Check if 'region_lengths' column exists (should be in both formats)
    if 'region_lengths' not in df.columns:
        raise ValueError(f"Column 'region_lengths' not found in {file_path}. Available columns: {df.columns.tolist()}")
    
    # Get region lengths and round them to integers
    region_lengths = df['region_lengths'].values
    region_lengths = np.rint(region_lengths).astype(int)
    
    # Calculate cumulative positions (change points are at region boundaries)
    cumsum_lengths = np.cumsum(region_lengths)
    
    # Change points are at the end of each region (subtract 1 for 0-based indexing)
    change_points = cumsum_lengths - 1
    
    # Return as a list (excluding the last point which is the end of the data)
    return change_points[:-1].tolist()

def read_data(data_file):
    """Read the signal data from CSV file."""
    with open(data_file, "r") as f:
        lines = f.readlines()[1:]  # Skip header
        data = np.array([int(line.strip().split(",")[1]) for line in lines])
    return data

def plot_change_points_overlay(data, detected_cps, true_cps, start_pos, end_pos, 
                                window_size, threshold, output_path):
    """Plot a section of data with detected and true change points overlaid."""
    plt.figure(figsize=(15, 6))
    
    # Plot the data
    positions = np.arange(start_pos, end_pos)
    data_section = data[start_pos:end_pos]
    plt.plot(positions, data_section, 'b-', linewidth=0.8, label='Data', alpha=0.7)
    
    # Plot true change points in the range
    true_in_range = [cp for cp in true_cps if start_pos <= cp <= end_pos]
    for cp in true_in_range:
        plt.axvline(x=cp, color=COLORS['green'], linestyle='--', linewidth=2, alpha=0.7)
    
    # Plot detected change points in the range
    detected_in_range = [cp for cp in detected_cps if start_pos <= cp <= end_pos]
    for cp in detected_in_range:
        plt.axvline(x=cp, color=COLORS['red'], linestyle='-', linewidth=1.5, alpha=0.7)
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLORS['blue'], linewidth=1, label='Signal'),
        Line2D([0], [0], color=COLORS['green'], linestyle='--', linewidth=2, label='True change points'),
        Line2D([0], [0], color=COLORS['red'], linestyle='-', linewidth=1.5, label='Detected change points')
    ]
    
    plt.xlabel('Position')
    plt.ylabel('Count')
    plt.title(f'Change Point Detection (Window={window_size}, Threshold={threshold:.2f})')
    plt.legend(handles=legend_elements, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_overlay_plots(dataset_name='pretty_data', num_plots=5, section_length=5000, 
                         base_results_folder=None):
    """Create overlay plots showing detected vs true change points on random sections of data."""
    
    # Setup paths
    data_file = f"Signal_processing/sample_data/{dataset_name}.csv"
    param_file = f"Signal_processing/sample_data/{dataset_name}_params.csv"
    
    # If base_results_folder not provided, try dataset-specific folder first, then fall back to general folder
    if base_results_folder is None:
        dataset_specific_folder = f"Signal_processing/results/sliding_mean_CPD/{dataset_name}"
        general_folder = "Signal_processing/results/sliding_mean_CPD"
        
        if os.path.exists(dataset_specific_folder):
            base_results_folder = dataset_specific_folder
        elif os.path.exists(general_folder):
            base_results_folder = general_folder
        else:
            print(f"Error: Could not find results folder for {dataset_name}")
            print(f"Tried: {dataset_specific_folder} and {general_folder}")
            return
    
    output_plots_folder = f"Signal_processing/results/overlay_plots/{dataset_name}"
    
    os.makedirs(output_plots_folder, exist_ok=True)
    
    # Read data and true change points
    data = read_data(data_file)
    true_cps = read_true_params(param_file)
    data_length = len(data)
    
    print(f"Data length: {data_length}")
    print(f"Number of true change points: {len(true_cps)}")
    
    # Get all window folders
    window_folders = [f for f in os.listdir(base_results_folder) 
                     if os.path.isdir(os.path.join(base_results_folder, f)) and f.startswith("window")]
    window_folders.sort()
    
    # For each window, select a few representative thresholds
    for window_folder in window_folders:
        window_size = int(window_folder.replace("window", ""))
        window_path = os.path.join(base_results_folder, window_folder)
        
        # Get all result files and select a few thresholds
        result_files = sorted([f for f in os.listdir(window_path) if f.endswith('.txt')])
        
        if len(result_files) == 0:
            continue
        
        # Select low, medium, high thresholds
        indices = [0, len(result_files)//2, -1]
        selected_files = [result_files[i] for i in indices if i < len(result_files)]
        
        for result_file in selected_files:
            # Extract threshold
            match = re.search(r'th(\d+\.\d+)', result_file)
            if not match:
                continue
            threshold = float(match.group(1))
            
            # Read detected change points
            file_path = os.path.join(window_path, result_file)
            detected_cps = read_change_points(file_path)
            
            print(f"Window {window_size}, Threshold {threshold:.2f}: {len(detected_cps)} detected CPs")
            
            # Generate random sections for overlay plots
            for plot_idx in range(num_plots):
                # Select a random starting position
                max_start = max(0, data_length - section_length)
                start_pos = np.random.randint(0, max_start + 1) if max_start > 0 else 0
                end_pos = min(start_pos + section_length, data_length)
                
                # Create output filename
                output_filename = f"overlay_ws{window_size}_th{threshold:.2f}_section{plot_idx+1}.png"
                output_path = os.path.join(output_plots_folder, output_filename)
                
                # Create the plot
                plot_change_points_overlay(data, detected_cps, true_cps, start_pos, end_pos,
                                          window_size, threshold, output_path)
            
            print(f"  Created {num_plots} overlay plots for window={window_size}, threshold={threshold:.2f}")
    
    print(f"\nOverlay plots saved to {output_plots_folder}")

def evaluate_all_windows_and_thresholds(dataset_name='pretty_data'):
    """Evaluate precision, recall, and F1 for all window sizes and thresholds.
    
    Args:
        dataset_name: Name of the dataset to evaluate (e.g., 'pretty_data', 'realistic_data')
    """

    # Setup paths
    true_param_file = f"Signal_processing/sample_data/{dataset_name}_params.csv"
    base_results_folder = f"Signal_processing/results/sliding_mean_CPD/{dataset_name}"
    output_csv = f"Signal_processing/results/performance_metrics_{dataset_name}.csv"
    output_plots_folder = f"Signal_processing/results/performance_plots/{dataset_name}"
    
    # Create output folder for plots
    os.makedirs(output_plots_folder, exist_ok=True)
    
    # Read true change points
    true_cps = read_true_params(true_param_file)
    print(f"Total true change points: {len(true_cps)}")
    
    # Collect all results
    results = []
    
    # Get all window folders
    window_folders = [f for f in os.listdir(base_results_folder) 
                     if os.path.isdir(os.path.join(base_results_folder, f)) and f.startswith("window")]
    window_folders.sort()
    
    print(f"Found window folders: {window_folders}")
    
    for window_folder in window_folders:
        # Extract window size from folder name
        window_size = int(window_folder.replace("window", ""))
        print(f"\nProcessing {window_folder} (window size: {window_size})...")
        
        # Define tolerances based on window size
        tolerances = {
            "full_window": window_size,
            "half_window": window_size // 2,
            "quarter_window": window_size // 4
        }
        
        window_path = os.path.join(base_results_folder, window_folder)
        
        # Get all result files in this window folder
        result_files = [f for f in os.listdir(window_path) if f.endswith('.txt')]
        result_files.sort()
        
        print(f"  Found {len(result_files)} result files")
        
        for result_file in result_files:
            # Extract threshold from filename using regex
            match = re.search(r'th(\d+\.\d+)', result_file)
            if not match:
                continue
            threshold = float(match.group(1))
            
            # Read detected change points
            file_path = os.path.join(window_path, result_file)
            detected_cps = read_change_points(file_path)
            
            # Calculate metrics for each tolerance
            for tol_name, tol_value in tolerances.items():
                prec = precision(detected_cps, true_cps, tol_value)
                rec = recall(detected_cps, true_cps, tol_value)
                f1 = F1_score(prec, rec)
                
                results.append({
                    'window_size': window_size,
                    'threshold': threshold,
                    'tolerance_type': tol_name,
                    'tolerance_value': tol_value,
                    'precision': prec,
                    'recall': rec,
                    'F1': f1,
                    'num_detected': len(detected_cps),
                    'num_true': len(true_cps)
                })
        
        print(f"  Processed {len(result_files)} files for {window_folder}")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")
    
    # Create precision-recall curves for each window and tolerance
    for tol_name in ["full_window", "half_window", "quarter_window"]:
        plt.figure(figsize=(10, 8))
        
        for window_folder in window_folders:
            window_size = int(window_folder.replace("window", ""))
            
            # Filter data for this window and tolerance
            window_data = results_df[
                (results_df['window_size'] == window_size) & 
                (results_df['tolerance_type'] == tol_name)
            ].sort_values('threshold')
            
            if len(window_data) > 0:
                plt.plot(window_data['recall'], window_data['precision'], 
                        marker='o', label=f'Window {window_size}', linewidth=2, markersize=4)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve (Tolerance: {tol_name})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1.05])
        plt.ylim([0, 1.05])
        
        plot_path = os.path.join(output_plots_folder, f'precision_recall_{tol_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {plot_path}")
    
    return results_df


if __name__ == "__main__":
    # Specify which dataset to analyze
    dataset_name = 'realistic_data'  # Change to 'realistic_data' to analyze the other dataset
    
    # Evaluate performance metrics
    print(f"Analyzing dataset: {dataset_name}")
    print("="*50)
    results_df = evaluate_all_windows_and_thresholds(dataset_name=dataset_name)
    print("\nSummary statistics:")
    print(results_df.groupby(['window_size', 'tolerance_type'])[['precision', 'recall', 'F1']].mean())
    
    # # Create overlay plots for visualization
    # print("\n" + "="*50)
    # print("Creating overlay plots...")
    # print("="*50)
    # create_overlay_plots(dataset_name=dataset_name, num_plots=3, section_length=5000)
    