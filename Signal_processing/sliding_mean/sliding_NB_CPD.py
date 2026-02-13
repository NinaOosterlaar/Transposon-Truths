import numpy as np
import os, sys
import matplotlib.pyplot as plt
import argparse
from scipy.special import gammaln

def nb_log_likelihood(x, mu, theta, eps=1e-10):
    x = np.asarray(x, dtype=np.float64)
    if np.any(x < 0):
        return -np.inf

    mu = np.clip(mu, eps, None)
    theta = np.clip(theta, eps, None)
    denom = np.clip(theta + mu, eps, None)

    t1 = gammaln(theta + x) - gammaln(theta) - gammaln(x + 1.0)
    t2 = theta * (np.log(theta) - np.log(denom))
    t3 = x * (np.log(mu) - np.log(denom))

    return np.sum(t1 + t2 + t3)

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


def sliding_NB_CPD(data, window_size, overlap, threshold):
    data = np.asarray(data, dtype=np.float64)
    step_size = int(window_size * (1 - overlap))
    n = len(data)
    change_points = []
    
    theta_global = fit_global_theta(data)
    last_cp = -np.inf
    last_score = 0
    
    
    return change_points


