import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from log_likelihoods import zinb_log_likelihood
from ZINB_MLE.EM import em_zinb_step
from ZINB_MLE.newton_raphson import newton_raphson_theta_step


def estimate_zinb(data, max_iter=100, tol=1e-6, tol_theta=1e-4, eps=1e-10, theta_max=1e6, theta_init_max=100):
    """
    Estimate ZINB parameters (pi, mu, theta) by iteratively running EM and Newton-Raphson.
    
    The algorithm:
    1. Initialize parameters using method of moments
    2. Iteratively:
       a. E-step: compute expectations (weights)
       b. M-step: update pi and mu
       c. Update theta using Newton-Raphson with the same weights
    3. Continue until convergence or max_iter reached
    
    Parameters:
    -----------
    data : array-like
        Observed count data
    max_iter : int
        Maximum number of iterations (EM + NR cycles)
    tol : float
        Convergence tolerance (parameter changes)
    eps : float
        Small value for numerical stability
    theta_max : float
        Maximum allowed value for theta
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'pi': final estimate of zero-inflation parameter
        - 'mu': final estimate of mean parameter
        - 'theta': final estimate of dispersion parameter
        - 'iterations': number of outer iterations performed
        - 'converged': boolean indicating if convergence was reached
        - 'log_likelihood': final log-likelihood
        - 'log_likelihood_history': list of log-likelihoods at each outer iteration
        - 'pi_history': list of pi values at each outer iteration
        - 'mu_history': list of mu values at each outer iteration
        - 'theta_history': list of theta values at each outer iteration
    """
    data = np.asarray(data, dtype=np.float64)
    N = len(data)
    
    # ===== INITIALIZATION =====
    pi = np.clip(np.mean(data == 0), eps, 1 - eps)

    ybar = np.mean(data)
    mu = np.clip(ybar / (1 - pi), eps, None)

    # crude: use var of all data to guess overdispersion of NB part
    v = np.var(data, ddof=1) if len(data) > 1 else 0.0

    # Under ZINB, Var(Y) = (1-pi)*(mu + mu^2/theta) + pi(1-pi)*mu^2
    # Solve approximately for theta (can go negative -> then set large)
    numer = (1 - pi) * mu**2
    denom = v - (1 - pi)*mu - pi*(1 - pi)*mu**2

    if denom > eps:
        theta = np.clip(numer / denom, eps, theta_init_max)
    else:
        theta = theta_init_max


        
    print(f"Initial parameters: pi={pi:.4f}, mu={mu:.4f}, theta={theta:.4f}")
    
    # Track convergence
    log_likelihood_history = []
    pi_history = [pi]
    mu_history = [mu]
    theta_history = [theta]
    converged = False
    
    # Compute initial log-likelihood
    ll = zinb_log_likelihood(data, mu, theta, pi, eps)
    log_likelihood_history.append(ll)
    
    # ===== ITERATIVE OPTIMIZATION =====
    for iteration in range(max_iter):
        # Store old parameters for convergence check
        pi_old = pi
        mu_old = mu
        theta_old = theta
        
        # ===== STEP 1: EM step (E-step + M-step to update pi, mu, and get weights) =====
        em_result = em_zinb_step(data, pi, mu, theta, eps=eps)
        
        pi = em_result['pi']
        mu = em_result['mu']
        weights = em_result['weights']  # a_i = P(z_i=0|y_i)
        
        # ===== STEP 2: Newton-Raphson step (update theta with weights from EM) =====
        for _ in range(20):
            theta_new = newton_raphson_theta_step(data, mu, weights, theta, eps=eps, theta_max=theta_max)
            if abs(theta_new - theta) / (theta + eps) < tol_theta:
                theta = theta_new
                break
            if theta >= theta_max * 0.999:
                theta = theta_max
                break
            theta = theta_new
        
        # Compute log-likelihood after this iteration
        ll = zinb_log_likelihood(data, mu, theta, pi, eps)
        log_likelihood_history.append(ll)
        
        # Store parameter history
        pi_history.append(pi)
        mu_history.append(mu)
        theta_history.append(theta)
        
        # ===== CHECK CONVERGENCE =====
        pi_change = abs(pi - pi_old)
        mu_change = abs(mu - mu_old) / (mu_old + eps)
        theta_change = abs(theta - theta_old) / (theta_old + eps)
        
        if pi_change < tol and mu_change < tol and theta_change < tol:
            converged = True
            break
    
    return {
        'pi': pi,
        'mu': mu,
        'theta': theta,
        'iterations': iteration + 1,
        'converged': converged,
        'log_likelihood': ll,
        'log_likelihood_history': log_likelihood_history,
        'pi_history': pi_history,
        'mu_history': mu_history,
        'theta_history': theta_history
    }

