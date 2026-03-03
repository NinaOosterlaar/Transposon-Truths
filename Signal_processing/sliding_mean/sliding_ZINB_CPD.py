import numpy as np
import os, sys
import matplotlib.pyplot as plt
import argparse
from concurrent.futures import ProcessPoolExecutor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from Signal_processing.ZINB_MLE.estimate_ZINB import estimate_zinb
from Signal_processing.ZINB_MLE.EM import em_zinb_step
from Signal_processing.log_likelihoods import zinb_log_likelihood

def sliding_ZINB_CPD(data, window_size, overlap, threshold, eps=1e-10, theta_global = None, tol = 1e-6, max_iter=10):
    data = np.asarray(data, dtype=np.float64)
    step_size = max(1, int(window_size * (1 - overlap)))
    n = len(data)
    
    if theta_global is None:
        theta_global = initialize_theta_global(data, eps=eps)
    
    change_points = []
    scores = []

    last_cp = -np.inf
    last_score = 0.0
    
    for start in range(0, n - 2 * window_size + 1, step_size):
        w1 = data[start : start + window_size]
        w2 = data[start + step_size : start + step_size + window_size]
        w0 = data[start : start + step_size + window_size]
        
        pi1 = np.clip(np.mean(w1 == 0), eps, 1 - eps)
        pi2 = np.clip(np.mean(w2 == 0), eps, 1 - eps)
        pi0 = np.clip(np.mean(w0 == 0), eps, 1 - eps)
        mu1 = np.clip(np.mean(w1) / (1 - pi1), eps, None)
        mu2 = np.clip(np.mean(w2) / (1 - pi2), eps, None)
        mu0 = np.clip(np.mean(w0) / (1 - pi0), eps, None)

        # Estimate parameters for w1 and w2 using EM with fixed theta_global
        for _ in range(max_iter):
            pi1_old, mu1_old = pi1, mu1
            result1 = em_zinb_step(w1, pi1, mu1, theta_global, eps=eps)
            pi1, mu1 = result1['pi'], result1['mu']
            if abs(pi1 - pi1_old) < tol and abs(mu1 - mu1_old) < tol:
                break
        for _ in range(max_iter):
            pi2_old, mu2_old = pi2, mu2
            result2 = em_zinb_step(w2, pi2, mu2, theta_global, eps=eps)
            pi2, mu2 = result2['pi'], result2['mu']
            if abs(pi2 - pi2_old) < tol and abs(mu2 - mu2_old) < tol:
                break
        for _ in range(max_iter):
            pi0_old, mu0_old = pi0, mu0
            result0 = em_zinb_step(w0, pi0, mu0, theta_global, eps=eps)
            pi0, mu0 = result0['pi'], result0['mu']
            if abs(pi0 - pi0_old) < tol and abs(mu0 - mu0_old) < tol:
                break


        ll1 = zinb_log_likelihood(w1, mu1, theta_global, pi1, eps=eps)
        ll2 = zinb_log_likelihood(w2, mu2, theta_global, pi2, eps=eps)
        ll0 = zinb_log_likelihood(w0, mu0, theta_global, pi0, eps=eps)

        glr = 2.0 * ((ll1 + ll2) - ll0)
        scores.append(glr)

        if glr > threshold:
            if (start - last_cp) >= window_size:  # Ensure minimum distance between change points
                change_points.append(start + window_size)  # CP is at the boundary between w1 and w2
                last_cp = start + window_size
                last_score = glr
            elif glr > last_score:  # If too close to last CP, keep the one with higher score
                change_points[-1] = start + window_size
                last_cp = start + window_size
                last_score = glr
        
    return change_points, scores
    
def initialize_theta_global(data, eps=1e-10, theta_max=1000):
    results = estimate_zinb(data, eps=eps)
    theta_global = results['theta']
    print(f"Estimated global theta: {theta_global:.4f}")
    print(f"(Estimated global pi: {results['pi']:.4f}, mu: {results['mu']:.4f})")
    if theta_global >= theta_max:
        # Throw an error that the estimation of theta failed
        raise ValueError("Estimated global theta is very large, indicating a failure in estimation. ")
    return theta_global

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
    

def process_window_size(ws, data, overlap, thresholds, theta_global, output_folder, dataset_name):
    """Process all thresholds for a given window size."""
    window_output_folder = os.path.join(output_folder, f"window{ws}")
    for threshold in thresholds:
        print(f"Processing window size: {ws}, threshold: {threshold:.2f}")
        change_points, scores = sliding_ZINB_CPD(data, ws, overlap, threshold, theta_global=theta_global)
        save_results(window_output_folder, dataset_name, change_points, scores, theta_global, ws, overlap, threshold)
    return ws

def parse_arguments():
    parser = argparse.ArgumentParser(description="Apply a sliding window mean change point detection algorithm on discrete count data.")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file containing the count data.")
    parser.add_argument("--output_folder", type=str, default="Signal_processing/results/sliding_mean/sliding_ZINB_CPD", help="Output folder for results.")
    parser.add_argument("--dataset_name", type=str, default="dataset", help="Name of the dataset being processed.")
    parser.add_argument("--n_workers", type=int, default=4, help="Number of parallel workers/CPUs to use.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    input_file = args.input_file
    window_size = [80, 10, 30, 50]
    overlap = 0.5
    thresholds = np.linspace(0.01, 10, 30)
    print(thresholds)
    output_folder = args.output_folder
    dataset_name = args.dataset_name
    # Add dataset name to output folder path
    output_folder = os.path.join(output_folder, dataset_name)
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Read data
    # datasets = read_csv_file_with_distances(input_file)
    with open(input_file, "r") as f:
        lines = f.readlines()[1:]  # Skip header
        data = [int(line.strip().split(",")[1]) for line in lines]
    theta_global = initialize_theta_global(data)
    print(f"Using global theta: {theta_global:.4f} for all window sizes and thresholds.")
    # theta_global = 1
    
    # Parallelize processing of different window sizes
    n_workers = min(args.n_workers, len(window_size))
    print(f"Using {n_workers} workers to process {len(window_size)} window sizes in parallel")
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(process_window_size, ws, data, overlap, thresholds, theta_global, output_folder, dataset_name)
            for ws in window_size
        ]
        for future in futures:
            ws_completed = future.result()
            print(f"Completed processing window size: {ws_completed}")
        
