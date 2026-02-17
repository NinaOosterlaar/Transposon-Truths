import numpy as np
import os, sys
import matplotlib.pyplot as plt
import argparse
from scipy.special import gammaln, logsumexp

def nb_logpmf(x, mu, theta, eps=1e-10):
    x = np.asarray(x, dtype=np.float64)
    if np.any(x < 0):
        return -np.inf

    mu = np.clip(mu, eps, None)
    theta = np.clip(theta, eps, None)
    denom = np.clip(theta + mu, eps, None)

    t1 = gammaln(theta + x) - gammaln(theta) - gammaln(x + 1.0)
    t2 = theta * (np.log(theta) - np.log(denom))
    t3 = x * (np.log(mu) - np.log(denom))

    return t1 + t2 + t3

def fit_global_theta(data, eps=1e-10, theta_max=1e8):
    x = np.asarray(data, dtype=np.float64)
    mu = float(np.mean(x))
    var = float(np.var(x, ddof=1)) if len(x) > 1 else 0.0

    if mu < eps:
        return theta_max  # all ~0 counts

    if var <= mu + eps:
        return theta_max  # ~Poisson / underdispersed -> very large theta

    theta = (mu * mu) / (var - mu)
    return float(np.clip(theta, eps, theta_max))

def zinb_log_likelihood(x, mu, theta, pi, eps=1e-10):
    x = np.asarray(x, dtype=np.float64)
    if np.any(x < 0):
        return -np.inf

    mu = float(np.clip(mu, eps, None))
    theta = float(np.clip(theta, eps, None))
    pi = float(np.clip(pi, eps, 1.0 - eps))

    log_nb = nb_logpmf(x, mu, theta, eps)  # elementwise log NB pmf

    is_zero = (x == 0)
    ll = np.empty_like(x, dtype=np.float64)

    # zeros: log( pi + (1-pi)*NB(0) ) stably
    if np.any(is_zero):
        a = np.log(pi)
        b = np.log(1.0 - pi) + log_nb[is_zero]
        ll[is_zero] = logsumexp(np.vstack([np.full(np.sum(is_zero), a), b]), axis=0)

    # positives: log(1-pi) + log NB(x)
    if np.any(~is_zero):
        ll[~is_zero] = np.log(1.0 - pi) + log_nb[~is_zero]

    return float(np.sum(ll))


def fit_ZINB_em(x, theta, max_iter=50, tol=1e-6, eps=1e-10):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    pi = float(np.clip(np.mean(x == 0) * 0.9, eps, 1.0 - eps))
    mu = float(np.clip(np.mean(x), eps, None))
    is_zero = (x == 0)

    prev_ll = -np.inf

    for _ in range(max_iter):
        n0 = int(np.sum(is_zero))

        # E-step: gamma = P(structural | x=0)
        if n0 > 0:
            a = np.log(pi)
            log_nb0 = nb_logpmf(np.array([0.0]), mu, theta, eps=eps)[0]
            b = np.log(1.0 - pi) + log_nb0
            gamma = float(np.exp(a - logsumexp([a, b])))
        else:
            gamma = 0.0

        # M-step: update pi
        pi_new = float(np.clip((n0 * gamma) / n, eps, 1.0 - eps))

        # M-step: update mu (NB-weighted mean)
        w = np.ones(n, dtype=np.float64)
        if n0 > 0:
            w[is_zero] = 1.0 - gamma
        mu_new = float(np.clip(np.sum(w * x) / np.sum(w), eps, None))

        # Check convergence by log-likelihood
        ll = zinb_log_likelihood(x, mu_new, theta, pi_new, eps=eps)
        if abs(ll - prev_ll) < tol:
            pi, mu = pi_new, mu_new
            break

        pi, mu = pi_new, mu_new
        prev_ll = ll

    return float(pi), float(mu)


def sliding_ZINB_CPD(data, window_size, overlap, threshold, theta_global=None, eps=1e-10):
    """
    Sliding-window GLR CPD under ZINB with global theta.
    H0: one (mu, pi) for both windows
    H1: separate (mu1, pi1) and (mu2, pi2)
    """
    data = np.asarray(data, dtype=np.float64)
    step_size = max(1, int(window_size * (1 - overlap)))
    n = len(data)

    if theta_global is None:
        theta_global = fit_global_theta(data, eps=eps)

    change_points = []
    scores = []
    params = []  # (pi1,mu1, pi2,mu2, pi0,mu0) per tested boundary

    last_cp = -np.inf
    last_score = -np.inf

    for start in range(0, n - 2 * window_size + 1, step_size):
        w1 = data[start : start + window_size]
        w2 = data[start + window_size : start + 2 * window_size]
        w0 = data[start : start + 2 * window_size]

        # Fit ZINB params in each region (theta fixed)
        pi1, mu1 = fit_ZINB_em(w1, theta_global, eps=eps)
        pi2, mu2 = fit_ZINB_em(w2, theta_global, eps=eps)
        pi0, mu0 = fit_ZINB_em(w0, theta_global, eps=eps)

        ll1 = zinb_log_likelihood(w1, mu1, theta_global, pi1, eps=eps)
        ll2 = zinb_log_likelihood(w2, mu2, theta_global, pi2, eps=eps)
        ll0 = zinb_log_likelihood(w0, mu0, theta_global, pi0, eps=eps)

        glr = 2.0 * ((ll1 + ll2) - ll0)
        scores.append(glr)
        params.append((pi1, mu1, pi2, mu2, pi0, mu0))

        cp_loc = start + window_size

        if glr > threshold:
            if (cp_loc - last_cp) >= window_size:
                change_points.append(cp_loc)
                last_cp = cp_loc
                last_score = glr
            elif glr > last_score:
                change_points[-1] = cp_loc
                last_cp = cp_loc
                last_score = glr

    return change_points, np.asarray(scores), theta_global, np.asarray(params, dtype=float)

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
        
