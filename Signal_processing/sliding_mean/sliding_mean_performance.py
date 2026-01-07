import numpy as np
import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import re
    

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
    """
    
    # Read the CSV file with region parameters
    df = pd.read_csv(file_path)
    
    # Get region lengths and round them to integers
    region_lengths = df['region_lengths'].values
    region_lengths = np.rint(region_lengths).astype(int)
    
    # Calculate cumulative positions (change points are at region boundaries)
    cumsum_lengths = np.cumsum(region_lengths)
    
    # Change points are at the end of each region (subtract 1 for 0-based indexing)
    change_points = cumsum_lengths - 1
    
    # Return as a list (excluding the last point which is the end of the data)
    return change_points[:-1].tolist()

def evaluate_all_windows_and_thresholds():
    """Evaluate precision, recall, and F1 for all window sizes and thresholds."""

    # Setup paths
    true_param_file = "Signal_processing/sample_data/pretty_data_params.csv"
    base_results_folder = "Signal_processing/results/sliding_mean_CPD"
    output_csv = "Signal_processing/results/performance_metrics.csv"
    output_plots_folder = "Signal_processing/results/performance_plots"
    
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
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve (Tolerance: {tol_name})', fontsize=14)
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
    results_df = evaluate_all_windows_and_thresholds()
    print("\nSummary statistics:")
    print(results_df.groupby(['window_size', 'tolerance_type'])[['precision', 'recall', 'F1']].mean())
    