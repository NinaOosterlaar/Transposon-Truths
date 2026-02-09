import torch

def l1_regularization(parameters):
    """
    Compute L1 regularization (sum of absolute values of all parameters).
    
    Parameters:
    -----------
    parameters : iterable of torch.Tensor
        Model parameters to regularize
    
    Returns:
    --------
    torch.Tensor
        Scalar L1 penalty
    """
    l1_penalty = 0.0
    for param in parameters:
        l1_penalty += torch.sum(torch.abs(param))
    return l1_penalty

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
    # Clamp inputs to safe ranges to prevent numerical issues
    theta = torch.clamp(theta, min=eps)
    mu    = torch.clamp(mu, min=eps)
    pi    = torch.clamp(pi, min=eps, max=1.0 - eps)
    
    # Check for NaN/Inf in inputs
    # if torch.isnan(mu).any() or torch.isinf(mu).any():
    #     print(f"WARNING: NaN/Inf detected in mu! min={mu.min()}, max={mu.max()}")
    # if torch.isnan(theta).any() or torch.isinf(theta).any():
    #     print(f"WARNING: NaN/Inf detected in theta! min={theta.min()}, max={theta.max()}")
    # if torch.isnan(pi).any() or torch.isinf(pi).any():
    #     print(f"WARNING: NaN/Inf detected in pi! min={pi.min()}, max={pi.max()}")

    # log NB pmf - use numerically stable computations
    # For lgamma, clamp inputs to prevent overflow
    t1 = (
        torch.lgamma(theta + x)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
    )
    t2 = theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
    t3 = x * (torch.log(mu + eps) - torch.log(theta + mu + eps))
    log_nb = t1 + t2 + t3
    
    # # Clamp log_nb to prevent extreme values
    # log_nb = torch.clamp(log_nb, min=-50, max=50)

    is_zero = (x == 0).float()

    # For zero-inflated component
    is_zero = (x == 0)

    log_pi = torch.log(pi)
    log_1_minus_pi = torch.log1p(-pi)

    log_prob_zero = torch.logaddexp(log_pi, log_1_minus_pi + log_nb)
    # log_prob_zero = log_pi
    log_prob_nonzero = log_1_minus_pi + log_nb

    log_prob = torch.where(is_zero, log_prob_zero, log_prob_nonzero)
    nll = -log_prob
    
    # Check for NaN/Inf in output
    # if torch.isnan(nll).any() or torch.isinf(nll).any():
    #     print(f"WARNING: NaN/Inf detected in NLL output!")
    #     print(f"  mu stats: min={mu.min():.4f}, max={mu.max():.4f}, mean={mu.mean():.4f}")
    #     print(f"  theta stats: min={theta.min():.4f}, max={theta.max():.4f}, mean={theta.mean():.4f}")
    #     print(f"  pi stats: min={pi.min():.4f}, max={pi.max():.4f}, mean={pi.mean():.4f}")
    #     # Replace NaN/Inf with large but finite value
    #     nll = torch.where(torch.isnan(nll) | torch.isinf(nll), torch.tensor(50.0, device=nll.device), nll)
    
    if reduction == 'sum':
        return nll.sum()
    elif reduction == 'mean':
        return nll.mean()
    elif reduction == 'none':
        return nll
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}. Choose 'sum', 'mean', or 'none'.")
    
def mae_loss(x, mu, pi, pi_threshold, reduction='sum'):
    """
    Mean Absolute Error loss between observed counts and mean parameter.
    
    Parameters:
    -----------
    x : torch.Tensor
        Observed counts (raw counts, not normalized)
    mu : torch.Tensor
        Mean parameter of NB distribution (after size factor correction)
    theta : torch.Tensor
        Dispersion parameter of NB distribution (not used in MAE)
    pi : torch.Tensor
        Zero-inflation probability (not used in MAE)
    eps : float
        Small constant for numerical stability (not used in MAE)
    reduction : str
        'sum', 'mean', or 'none'. Default='sum' for consistency with PyTorch losses
        
    Returns:
    --------
    torch.Tensor
        MAE loss. Shape depends on reduction:
        - 'sum': scalar (sum over all elements)
        - 'mean': scalar (mean over all elements)
        - 'none': same shape as input (per-element loss)
    """
    reconstruction = mu * (pi < pi_threshold).float()
    mae = torch.abs(x - reconstruction)
    
    if reduction == 'sum':
        return mae.sum()
    elif reduction == 'mean':
        return mae.mean()
    elif reduction == 'none':
        return mae
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}. Choose 'sum', 'mean', or 'none'.")
    
    
def segment_transition_loss(mu, pi, eps=1e-8):
    """
    Encourage sharp segmentation by penalizing gradual transitions.
    
    This loss promotes piecewise-constant parameters (mu, pi) by:
    1. Allowing no change (same segment) -> low penalty
    2. Allowing large changes (sharp boundary) -> penalty saturates
    3. Penalizing medium/gradual changes -> high penalty (forces commitment)
    
    Uses a non-convex potential x/(1+x) that:
    - Is ~0 for small differences (stable segments)
    - Grows linearly for medium differences (penalizes gradual transitions)
    - Saturates at 1 for large differences (sharp boundaries OK)
    
    Parameters:
    -----------
    mu : torch.Tensor
        Mean parameter of NB distribution, shape (batch, seq_length)
    pi : torch.Tensor
        Zero-inflation probability, shape (batch, seq_length)
    eps : float
        Small constant for numerical stability
    
    Returns:
    --------
    torch.Tensor
        Scalar transition penalty (encourages sharp boundaries)
    """
    # Work in log/logit space without normalization to avoid numerical issues
    # mu: use log scale (counts are multiplicative)
    mu_safe = torch.clamp(mu, min=eps, max=1e10)  # Prevent explosion
    log_mu = torch.log(mu_safe)
    
    # pi: use logit scale (probabilities are on logit scale)
    pi_safe = torch.clamp(pi, min=eps, max=1.0 - eps)
    logit_pi = torch.log(pi_safe / (1.0 - pi_safe))
    
    # Compute differences between consecutive positions (already in comparable scales)
    # Scale log_mu differences to be comparable to logit_pi (typically in [-5, 5] range)
    log_mu_diff = torch.abs(log_mu[:, 1:] - log_mu[:, :-1])  # (batch, seq-1)
    logit_pi_diff = torch.abs(logit_pi[:, 1:] - logit_pi[:, :-1])
    
    # Combined distance in (mu, pi) space - simple L2 norm
    combined_diff = torch.sqrt(log_mu_diff**2 + logit_pi_diff**2 + eps)
    
    # Non-convex penalty: x / (1 + x)
    # This saturates at 1.0 for large changes (boundaries OK)
    # But grows linearly for small-to-medium changes (penalizes gradual transitions)
    penalty = combined_diff / (1.0 + combined_diff)
    
    return penalty.mean()


def reconstruct_masked_values(x, mu, pi, mask, pi_threshold):
    """
    Loss that measures how well the model reconstructs masked values to not be zero.
    
    Parameters:
    -----------
    x : torch.Tensor
        Observed counts (raw counts, not normalized)
    mu : torch.Tensor
        Mean parameter of NB distribution (after size factor correction)
    pi : torch.Tensor
        Zero-inflation probability (between 0 and 1)
    mask : torch.Tensor
        Boolean mask indicating which values to reconstruct (True = reconstruct)
    pi_threshold : float
        Threshold for zero-inflation probability to consider a value as non-zero
    
    Returns:
    --------
    torch.Tensor
        Tensor with masked values reconstructed.
    """
    masked_values = x[mask]
    if masked_values.numel() == 0:
        return torch.tensor(0.0, device=x.device)
    reconstruction = mu[mask] * (pi[mask] < pi_threshold).float()
    loss = torch.abs(masked_values - reconstruction)
    return loss.mean()

