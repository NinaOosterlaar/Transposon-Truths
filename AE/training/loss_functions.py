import torch

def zinb_nll(x, mu, theta, pi, eps=1e-8, reduction='sum'):
    """
    Zero-Inflated Negative Binomial Negative Log-Likelihood loss.
    Numerically stable version with additional safeguards.
    
    Parameters:
    -----------
    x : torch.Tensor
        Observed counts (raw counts, not normalized)
    mu : torch.Tensor
        Mean parameter of NB distribution (after size factor correction)
    theta : torch.Tensor
        Dispersion parameter of NB distribution (positive)
    pi : torch.Tensor
        Zero-inflation probability (between 0 and 1)
    eps : float
        Small constant for numerical stability
    reduction : str
        'sum', 'mean', or 'none'. Default='sum' for consistency with PyTorch losses
    
    Returns:
    --------
    torch.Tensor
        Negative log-likelihood. Shape depends on reduction:
        - 'sum': scalar (sum over all elements)
        - 'mean': scalar (mean over all elements)
        - 'none': same shape as input (per-element loss)
    """
    # Clamp inputs to safe ranges to prevent numerical issues
    theta = torch.clamp(theta, min=eps, max=1e6)
    mu    = torch.clamp(mu,    min=eps, max=1e6)
    pi    = torch.clamp(pi,    min=eps, max=1.0 - eps)
    
    # Check for NaN/Inf in inputs
    if torch.isnan(mu).any() or torch.isinf(mu).any():
        print(f"WARNING: NaN/Inf detected in mu! min={mu.min()}, max={mu.max()}")
    if torch.isnan(theta).any() or torch.isinf(theta).any():
        print(f"WARNING: NaN/Inf detected in theta! min={theta.min()}, max={theta.max()}")
    if torch.isnan(pi).any() or torch.isinf(pi).any():
        print(f"WARNING: NaN/Inf detected in pi! min={pi.min()}, max={pi.max()}")

    # log NB pmf - use numerically stable computations
    # For lgamma, clamp inputs to prevent overflow
    t1 = (
        torch.lgamma(torch.clamp(theta + x, max=1e6))
        - torch.lgamma(torch.clamp(theta, max=1e6))
        - torch.lgamma(torch.clamp(x + 1.0, max=1e6))
    )
    t2 = theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
    t3 = x * (torch.log(mu + eps) - torch.log(theta + mu + eps))
    log_nb = t1 + t2 + t3
    
    # Clamp log_nb to prevent extreme values
    log_nb = torch.clamp(log_nb, min=-50, max=50)

    is_zero = (x < eps).float()

    # For zero-inflated component
    # log(pi + (1-pi) * exp(log_nb)) - use log-sum-exp trick for stability
    log_pi = torch.log(pi + eps)
    log_1_minus_pi = torch.log(1.0 - pi + eps)
    
    # log-sum-exp trick: log(a + b) = log(a) + log(1 + exp(log(b) - log(a)))
    # For zeros: log(pi + (1-pi)*exp(log_nb))
    log_prob_zero_nb = log_1_minus_pi + log_nb
    log_prob_zero = torch.where(
        log_pi > log_prob_zero_nb,
        log_pi + torch.log(1.0 + torch.exp(log_prob_zero_nb - log_pi) + eps),
        log_prob_zero_nb + torch.log(torch.exp(log_pi - log_prob_zero_nb) + 1.0 + eps)
    )
    
    # For non-zeros: log((1-pi) * exp(log_nb)) = log(1-pi) + log_nb
    log_prob_nonzero = log_1_minus_pi + log_nb

    log_prob = is_zero * log_prob_zero + (1.0 - is_zero) * log_prob_nonzero
    
    # Clamp final log_prob to prevent extreme values
    log_prob = torch.clamp(log_prob, min=-50, max=50)
    
    nll = -log_prob
    
    # Check for NaN/Inf in output
    if torch.isnan(nll).any() or torch.isinf(nll).any():
        print(f"WARNING: NaN/Inf detected in NLL output!")
        print(f"  mu stats: min={mu.min():.4f}, max={mu.max():.4f}, mean={mu.mean():.4f}")
        print(f"  theta stats: min={theta.min():.4f}, max={theta.max():.4f}, mean={theta.mean():.4f}")
        print(f"  pi stats: min={pi.min():.4f}, max={pi.max():.4f}, mean={pi.mean():.4f}")
        # Replace NaN/Inf with large but finite value
        nll = torch.where(torch.isnan(nll) | torch.isinf(nll), torch.tensor(50.0, device=nll.device), nll)
    
    if reduction == 'sum':
        return nll.sum()
    elif reduction == 'mean':
        return nll.mean()
    elif reduction == 'none':
        return nll
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}. Choose 'sum', 'mean', or 'none'.")