import torch

def zinb_nll(x, mu, theta, pi, eps=1e-8, reduction='sum'):
    """
    Zero-Inflated Negative Binomial Negative Log-Likelihood loss.
    
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
    theta = torch.clamp(theta, min=eps)
    mu    = torch.clamp(mu,    min=eps)
    pi    = torch.clamp(pi,    min=eps, max=1.0 - eps)

    # log NB pmf
    t1 = (
        torch.lgamma(theta + x)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
    )
    t2 = theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
    t3 = x * (torch.log(mu + eps) - torch.log(theta + mu + eps))
    log_nb = t1 + t2 + t3

    is_zero = (x < eps).float()

    log_prob_zero = torch.log(
        pi + (1.0 - pi) * torch.exp(log_nb) + eps
    )
    log_prob_nonzero = torch.log(1.0 - pi + eps) + log_nb

    log_prob = is_zero * log_prob_zero + (1.0 - is_zero) * log_prob_nonzero
    
    nll = -log_prob
    
    if reduction == 'sum':
        return nll.sum()
    elif reduction == 'mean':
        return nll.mean()
    elif reduction == 'none':
        return nll
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}. Choose 'sum', 'mean', or 'none'.")