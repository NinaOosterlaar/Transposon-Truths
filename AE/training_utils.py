import torch
import torch.nn as nn

def add_noise(y, denoise_percent):
    """
    Randomly set a percentage of non-zero values to zero for denoising autoencoder.
    
    Parameters:
    -----------
    y : torch.Tensor
        Input tensor (B, seq)
    denoise_percent : float
        Percentage of non-zero values to set to zero (0.0 to 1.0)
    
    Returns:
    --------
    y_noisy : torch.Tensor
        Noisy version of input with some non-zero values set to zero
    """
    if denoise_percent <= 0.0:
        return y, torch.zeros_like(y, dtype=torch.bool)

    y_noisy = y.clone()
    mask = torch.zeros_like(y, dtype=torch.bool)

    B = y_noisy.size(0)

    for b in range(B):
        nz = torch.nonzero(y_noisy[b] != 0, as_tuple=True)[0]
        num_non_zero = nz.numel()
        if num_non_zero == 0:
            continue

        num_to_zero = int(num_non_zero * denoise_percent)
        if num_to_zero <= 0:
            continue

        perm = torch.randperm(num_non_zero, device=y_noisy.device)[:num_to_zero]
        seq_idx = nz[perm]

        y_noisy[b, seq_idx] = 0
        mask[b, seq_idx] = True

    return y_noisy, mask


# Embed the chromosome feature if needed
class ChromosomeEmbedding(nn.Module):
    def __init__(self, num_chromosomes=17, embedding_dim=4):
        super(ChromosomeEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_chromosomes, embedding_dim)
        
    def forward(self, x):
        return self.embedding(x)

def gaussian_kl(mu, logvar):
    logvar = torch.clamp(logvar, min=-20, max=10)
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def dataloader_from_array(input, chrom=True, batch_size=64, shuffle=True, binary=False, zinb=False):
    data_array = np.load(input)
    counts = data_array[:, :, 0]  # Normalized counts (Value)
    
    if binary:
        # Create binary targets based on counts > 0
        y_binary = (counts > 0).astype(np.float32)
    
    # For ZINB mode: extract Value_Raw and Size_Factor from the end of the array
    if zinb:
        if chrom:
            # Structure: [Value, features..., Chrom, Value_Raw, Size_Factor]
            features = data_array[:, :, 1:-3]  # Exclude Value, Chrom, Value_Raw, Size_Factor
            chrom_indices = data_array[:, :, -3].astype(np.int64)
            raw_counts = data_array[:, :, -2]  # Value_Raw
            size_factors = data_array[:, 0, -1]  # Size_Factor (take first position, all same)
        else:
            # Structure: [Value, features..., Value_Raw, Size_Factor]
            features = data_array[:, :, 1:-2]  # Exclude Value, Value_Raw, Size_Factor
            raw_counts = data_array[:, :, -2]  # Value_Raw
            size_factors = data_array[:, 0, -1]  # Size_Factor (take first position, all same)
    else:
        # Standard mode (no ZINB)
        if chrom:
            features = data_array[:, :, 1:-1]
            chrom_indices = data_array[:, :, -1].astype(np.int64)
        else:
            features = data_array[:, :, 1:]
    
    x_tensor = torch.tensor(features, dtype=torch.float32)
    y_tensor = torch.tensor(counts, dtype=torch.float32)
    
    if binary:
        y_binary_tensor = torch.tensor(y_binary, dtype=torch.float32)
    
    # Build dataset tensors
    tensors = [x_tensor, y_tensor]
    
    if binary:
        tensors.append(y_binary_tensor)
    
    if chrom:
        c_tensor = torch.tensor(chrom_indices, dtype=torch.long)
        tensors.append(c_tensor)
    
    if zinb:
        y_raw_tensor = torch.tensor(raw_counts, dtype=torch.float32)
        sf_tensor = torch.tensor(size_factors, dtype=torch.float32)
        tensors.append(y_raw_tensor)
        tensors.append(sf_tensor)
    
    dataset = torch.utils.data.TensorDataset(*tensors)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader