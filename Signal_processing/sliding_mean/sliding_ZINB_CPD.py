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