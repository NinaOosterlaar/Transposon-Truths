import numpy as np
import os, sys
import matplotlib.pyplot as plt
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from Signal_processing.ZINB_MLE.estimate_ZINB import estimate_zinb
from Signal_processing.ZINB_MLE.EM import em_zinb_step
from Signal_processing.log_likelihoods import zinb_log_likelihood



def save_results(output_folder, dataset_name, change_points, scores, theta_global, window_size, overlap, threshold):  
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file = os.path.join(output_folder, f"{dataset_name}_ws{window_size}_ov{int(overlap*100)}_th{threshold:.2f}.txt")  
    with open(output_file, "w") as f:
        for cp in change_points:
            f.write(f"{cp} \n")
        f.write(f"scores: {scores}\n")
        f.write(f"theta_global: {theta_global}\n")
        f.write(f"window_size: {window_size}, overlap: {overlap}, threshold: {threshold}\n")
    

def parse_arguments():
    parser = argparse.ArgumentParser(description="Apply a sliding window mean change point detection algorithm on discrete count data.")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file containing the count data.")
    parser.add_argument("--output_folder", type=str, default="Signal_processing/results/sliding_mean/sliding_NB_CPD", help="Output folder for results.")
    parser.add_argument("--dataset_name", type=str, default="dataset", help="Name of the dataset being processed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    input_file = args.input_file
    window_size = [80]
    overlap = 0.5
    thresholds = np.linspace(0.5, 25, 30)
    print(thresholds)
    output_folder = args.output_folder
    dataset_name = args.dataset_name
    # Add dataset name to output folder path
    output_folder = os.path.join(output_folder, dataset_name)
    
    # Read data
    # datasets = read_csv_file_with_distances(input_file)
    with open(input_file, "r") as f:
        lines = f.readlines()[1:]  # Skip header
        data = [int(line.strip().split(",")[1]) for line in lines]
    for ws in window_size:
        window_output_folder = os.path.join(output_folder, f"window{ws}")
        for threshold in thresholds:
            print(f"Processing window size: {ws}, threshold: {threshold:.2f}")
            change_points, scores, theta_global, params = sliding_ZINB_CPD(data, ws, overlap, threshold)
            save_results(window_output_folder, dataset_name, change_points, scores, theta_global, ws, overlap, threshold, params)
        
