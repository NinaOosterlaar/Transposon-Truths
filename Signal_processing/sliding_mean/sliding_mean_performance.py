import numpy as np
import os, sys

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
    """Read true change point parameters from a CSV file."""
    true_params = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            dataset_name = parts[0]
            cps = list(map(int, parts[1:]))
            true_params[dataset_name] = cps
    return true_params

if __name__ == "__main__":
    true_param_file = "Signal_processing/sample_data/pretty_data_params.csv"
    change_points_folder = "Signal_processing/results/sliding_mean_CPD/"
    change_points = read_change_points(os.path.join(change_points_folder, "pretty_data_ws30_ov50_th3.txt"))
    print(f"Detected change points: {len(change_points)}")
    
    