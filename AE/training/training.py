import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, r2_score
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from AE.plotting.plot_loss import plot_zinb_training_loss
from AE.plotting.results_ZINB import plot_zinb_test_results
import argparse
from AE.architectures.ZINBAE import ZINBAE, ZINBVAE
from AE.training.loss_functions import zinb_nll, l1_regularization, reconstruct_masked_values
from AE.training.training_utils import ChromosomeEmbedding, dataloader_from_array, gaussian_kl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def train(model, dataloader, num_epochs=50, learning_rate=1e-3, chrom=True, chrom_embedding=None, plot=True, beta=1.0, name="", denoise_percent=0.3, regularizer='none', alpha=0.0, gamma=0.0, pi_threshold=0.5):
    """
    Train ZINBAE or ZINBVAE model
    
    Parameters:
    -----------
    model : ZINBAE or ZINBVAE
        The model to train
    dataloader : DataLoader
        Training data
    num_epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate for optimizer
    chrom : bool
        Whether to use chromosome embedding
    chrom_embedding : ChromosomeEmbedding or None
        Chromosome embedding module (created externally to ensure consistency)
    plot : bool
        Whether to plot training loss
    beta : float
        Weight for KL divergence loss (only used for ZINBVAE). Default=1.0
    denoise_percent : float
        Percentage of non-zero values that were masked in the dataloader (0.0 to 1.0). 
        Note: Masking is now applied once in dataloader_from_array(), not per epoch. Default=0.0
    regularizer : str
        Type of regularization: 'none', 'L1', or 'L2'. Default='none'
    alpha : float
        Regularization strength. Default=0.0
    gamma : float
        Weight for masked reconstruction loss. Default=0.0
    pi_threshold : float
        Threshold for zero-inflation probability to consider a value as non-zero. Default=0.5
    """
    model.to(device)
    parameters = list(model.parameters())
    if chrom:
        if chrom_embedding is None:
            raise ValueError("chrom_embedding must be provided when chrom=True")
        chrom_embedding.to(device)
        parameters += list(chrom_embedding.parameters())

    # Determine model type (ZINBAE or ZINBVAE)
    is_zinbvae = getattr(model, "model_type", None) == "ZINBVAE"
    
    # Create optimizer with L2 regularization if specified
    weight_decay = alpha if regularizer.lower() == 'l2' else 0.0
    optimizer = optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    
    epoch_losses = []
    epoch_recon_losses = []  # For ZINBVAE/VAE
    epoch_kl_losses = []      # For ZINBVAE/VAE
    epoch_reg_losses = []     # For regularization
    epoch_masked_losses = []  # For masked reconstruction loss
    
    # Collect masks from dataloader for evaluation
    training_masks_collected = []
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_reg_loss = 0.0
        epoch_masked_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            # Unpack batch for ZINB models
            # Note: mask is always the last element in batc
            if chrom:
                x, y, c, y_raw, size_factors, mask = batch
                c = c.to(device)
            else:
                x, y, y_raw, size_factors, mask = batch
            
            x = x.to(device)         # (B, seq, F_other)
            y = y.to(device)         # (B, seq) - normalized counts (already masked if denoise_percent > 0)
            y_raw = y_raw.to(device)  # (B, seq) - raw counts (unmasked/original)
            size_factors = size_factors.to(device)  # (B,) - size factors
            mask = mask.to(device)   # (B, seq) - boolean mask indicating masked positions
            
            optimizer.zero_grad()
            
            # Collect masks (fixed from dataloader, same across all epochs)
            if denoise_percent > 0 and epoch == 0:  # Only collect once in first epoch
                training_masks_collected.append(mask.detach().cpu().numpy())
            
            # y is already masked (if denoise_percent > 0), use it directly
            y_in = y.unsqueeze(-1)  # Add feature dimension
            if chrom:
                c_emb = chrom_embedding(c)
                batch_input = torch.cat((y_in, x, c_emb), dim=2)
            else:
                batch_input = torch.cat((y_in, x), dim=2)

            # reset per batch
            recon_loss = None
            kl_loss = None
            masked_loss = None

            # Forward pass for ZINB models
            if is_zinbvae:
                mu, theta, pi, z, mu_z, logvar_z = model(batch_input, size_factors)
                recon_loss = zinb_nll(y_raw, mu, theta, pi, reduction='mean')
                # KL loss: divide by seq_length to get "per-element" KL
                kl_loss = gaussian_kl(mu_z, logvar_z) / model.seq_length
                loss = recon_loss + beta * kl_loss
                # Add masked reconstruction loss if gamma > 0
                if gamma > 0 and denoise_percent > 0:
                    masked_loss = reconstruct_masked_values(y_raw, mu, pi, mask, pi_threshold)
                    loss = loss + gamma * masked_loss
            else:  # ZINBAE
                mu, theta, pi, z = model(batch_input, size_factors)
                recon_loss = zinb_nll(y_raw, mu, theta, pi, reduction='mean')
                loss = recon_loss
                # Add masked reconstruction loss if gamma > 0
                if gamma > 0 and denoise_percent > 0:
                    masked_loss = reconstruct_masked_values(y_raw, mu, pi, mask, pi_threshold)
                    loss = loss + gamma * masked_loss
                kl_loss = None

            # Add L1 regularization if specified
            reg_penalty = 0.0
            if regularizer.lower() == 'l1' and alpha > 0:
                l1_penalty = l1_regularization(parameters)
                reg_penalty = alpha * l1_penalty.item()
                loss = loss + alpha * l1_penalty
            elif regularizer.lower() == 'l2' and alpha > 0:
                # L2 penalty for tracking (already applied via weight_decay in optimizer)
                l2_penalty = sum(torch.sum(p ** 2) for p in parameters)
                reg_penalty = alpha * l2_penalty.item()

            # bookkeeping (recon_loss and kl_loss are already per-sample averages)
            epoch_recon_loss += recon_loss.item() * y.size(0)
            if kl_loss is not None:
                epoch_kl_loss += kl_loss.item() * y.size(0)
            if masked_loss is not None:
                epoch_masked_loss += masked_loss.item() * y.size(0)
            epoch_reg_loss += reg_penalty * y.size(0)
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
            
            optimizer.step()
            
            # Check for NaN in loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nWARNING: NaN or Inf detected in loss at epoch {epoch+1}!")
                print(f"Batch info: batch size={y.size(0)}")
                print(f"  mu: min={mu.min():.4f}, max={mu.max():.4f}, mean={mu.mean():.4f}")
                print(f"  theta: min={theta.min():.4f}, max={theta.max():.4f}, mean={theta.mean():.4f}")
                print(f"  pi: min={pi.min():.4f}, max={pi.max():.4f}, mean={pi.mean():.4f}")
                if is_zinbvae:
                    print(f"  mu_z: min={mu_z.min():.4f}, max={mu_z.max():.4f}, mean={mu_z.mean():.4f}")
                    print(f"  logvar_z: min={logvar_z.min():.4f}, max={logvar_z.max():.4f}, mean={logvar_z.mean():.4f}")
                # Skip this batch or break
                continue
            
            epoch_loss += loss.item() * y.size(0)
            
            # Update progress bar with current loss
            if is_zinbvae:
                postfix_dict = {
                    'total': f'{loss.item():.4f}',
                    'recon': f'{recon_loss.item():.4f}',
                    'kl': f'{kl_loss.item():.4f}'
                }
                if masked_loss is not None:
                    postfix_dict['masked'] = f'{masked_loss.item():.4f}'
                if regularizer.lower() != 'none' and alpha > 0:
                    postfix_dict['reg'] = f'{reg_penalty:.6f}'
                pbar.set_postfix(postfix_dict)
            else:  # ZINBAE
                postfix_dict = {
                    'total': f'{loss.item():.4f}',
                    'nll': f'{recon_loss.item():.4f}'
                }
                if masked_loss is not None:
                    postfix_dict['masked'] = f'{masked_loss.item():.4f}'
                    postfix_dict['reg'] = f'{reg_penalty:.6f}'
                pbar.set_postfix(postfix_dict)
        
        epoch_loss /= len(dataloader.dataset)
        epoch_losses.append(epoch_loss)
        
        # Track regularization loss
        epoch_reg_loss /= len(dataloader.dataset)
        epoch_reg_losses.append(epoch_reg_loss)
        
        # Track masked reconstruction loss
        epoch_masked_loss /= len(dataloader.dataset)
        epoch_masked_losses.append(epoch_masked_loss)
        
        # Track recon and KL losses
        epoch_recon_loss /= len(dataloader.dataset)
        epoch_recon_losses.append(epoch_recon_loss)
        
        if is_zinbvae:
            epoch_kl_loss /= len(dataloader.dataset)
            epoch_kl_losses.append(epoch_kl_loss)
            # Build print message with optional components
            msg = f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {epoch_loss:.4f}, Recon: {epoch_recon_loss:.4f}, KL: {epoch_kl_loss:.4f}"
            if gamma > 0 and denoise_percent > 0:
                msg += f", Masked: {epoch_masked_loss:.4f}"
            if regularizer.lower() != 'none' and alpha > 0:
                msg += f", Reg: {epoch_reg_loss:.6f}"
            print(msg)
        else:  # ZINBAE
            # Build print message with optional components
            msg = f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {epoch_loss:.4f}, NLL: {epoch_recon_loss:.4f}"
            if gamma > 0 and denoise_percent > 0:
                msg += f", Masked: {epoch_masked_loss:.4f}"
            if regularizer.lower() != 'none' and alpha > 0:
                msg += f", Reg: {epoch_reg_loss:.6f}"
            print(msg)
    
    if plot:
        # Pass regularization losses if regularization is active
        reg_losses_to_plot = epoch_reg_losses if (regularizer.lower() != 'none' and alpha > 0) else None
        # Pass masked losses if gamma > 0
        masked_losses_to_plot = epoch_masked_losses if (gamma > 0 and denoise_percent > 0) else None
        
        model_type_str = model.model_type
        use_conv = model.use_conv if hasattr(model, 'use_conv') else False
        
        if is_zinbvae:
            plot_zinb_training_loss(epoch_losses, epoch_recon_losses, epoch_kl_losses, 
                                   model_type=model_type_str, use_conv=use_conv, name=name,
                                   reg_losses=reg_losses_to_plot, masked_losses=masked_losses_to_plot)
        else:  # ZINBAE
            plot_zinb_training_loss(epoch_losses, epoch_recon_losses, None,
                                   model_type=model_type_str, use_conv=use_conv, name=name,
                                   reg_losses=reg_losses_to_plot, masked_losses=masked_losses_to_plot)
    
    # Evaluate on training data to get reconstruction plots
    print("\n" + "="*50)
    print("EVALUATING ON TRAINING DATA")
    print("="*50)
    
    test(model, dataloader, chrom=chrom, chrom_embedding=chrom_embedding, 
         plot=True, n_examples=5, beta=beta, name=name, 
         denoise_percent=denoise_percent, eval_mode="training", threshold=pi_threshold,
         gamma=gamma, pi_threshold=pi_threshold, regularizer=regularizer, alpha=alpha)
    
    return model

def test(model, dataloader, chrom=True, chrom_embedding=None, plot=True, n_examples=5, beta=1.0, name="", denoise_percent=0.0, eval_mode="testing", threshold=0.5, gamma=0.0, pi_threshold=0.5, regularizer='none', alpha=0.0):
    """
    Test ZINBAE or ZINBVAE model
    
    Parameters:
    -----------
    model : ZINBAE or ZINBVAE
        The model to test
    dataloader : DataLoader
        Test data
    chrom : bool
        Whether to use chromosome embedding
    chrom_embedding : ChromosomeEmbedding or None
        Chromosome embedding module (must be the same one used during training)
    plot : bool
        Whether to create visualization plots
    n_examples : int
        Number of example reconstructions to plot
    beta : float
        Weight for KL divergence loss (only used for ZINBVAE). Default=1.0
    denoise_percent : float
        Percentage of non-zero values that were masked in the dataloader (0.0 to 1.0).
    eval_mode : str
        Either "testing" or "training" - determines subdirectory for saving plots. Default="testing"
    threshold : float
        Threshold for zero-inflation probability. Default=0.5
    gamma : float
        Weight for masked reconstruction loss. Default=0.0
    pi_threshold : float
        Threshold for zero-inflation probability for masked loss. Default=0.5
    regularizer : str
        Type of regularization: 'none', 'L1', or 'L2'. Default='none'
    alpha : float
        Regularization strength. Default=0.0
    """
    model.to(device)
    model.eval()
    
    if chrom:
        if chrom_embedding is None:
            raise ValueError("chrom_embedding must be provided when chrom=True")
        chrom_embedding.to(device)
        chrom_embedding.eval()
    
    # Determine model type (ZINBAE or ZINBVAE)
    is_zinbvae = getattr(model, "model_type", None) == "ZINBVAE"
    
    all_reconstructions = []
    all_latents = []
    all_originals = []
    all_theta = []
    all_pi = []
    all_raw_counts = []
    all_mu_raw = []  # unmasked mu values for plotting
    all_masks = []  # Store masks for denoising analysis
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_masked_loss = 0.0
    total_reg_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            # Unpack batch for ZINB models
            # Note: mask is always the last element in batch
            if chrom:
                x, y, c, y_raw, size_factors, mask = batch
                c = c.to(device)
                x = x.to(device)
            else:
                x, y, y_raw, size_factors, mask = batch
                x = x.to(device)
            
            y = y.to(device)         # (B, seq) - normalized counts (already masked if denoise_percent > 0)
            y_raw = y_raw.to(device)  # (B, seq) - raw counts (unmasked/original)
            size_factors = size_factors.to(device)  # (B,) - size factors
            mask = mask.to(device)   # (B, seq) - boolean mask indicating masked positions
            
            # y is already masked (if denoise_percent > 0), use it directly
            y_in = y.unsqueeze(-1)  # Add feature dimension
            if chrom:
                c_emb = chrom_embedding(c)
                batch_input = torch.cat((y_in, x, c_emb), dim=2)
            else:
                batch_input = torch.cat((y_in, x), dim=2)

            batch_size = y.size(0)

            # Forward pass for ZINB models
            if is_zinbvae:
                mu, theta, pi, z, mu_z, logvar_z = model(batch_input, size_factors)
                recon_loss = zinb_nll(y_raw, mu, theta, pi, reduction="mean")
                # KL loss: divide by seq_length to get "per-element" KL
                kl_loss = gaussian_kl(mu_z, logvar_z) / model.seq_length
                loss = recon_loss + beta * kl_loss
                # Add masked reconstruction loss if gamma > 0
                if gamma > 0 and denoise_percent > 0:
                    masked_loss = reconstruct_masked_values(y_raw, mu, pi, mask, pi_threshold)
                    loss = loss + gamma * masked_loss
            else:  # ZINBAE
                mu, theta, pi, z = model(batch_input, size_factors)
                recon_loss = zinb_nll(y_raw, mu, theta, pi, reduction="mean")
                loss = recon_loss
                # Add masked reconstruction loss if gamma > 0
                if gamma > 0 and denoise_percent > 0:
                    masked_loss = reconstruct_masked_values(y_raw, mu, pi, mask, pi_threshold)
                    loss = loss + gamma * masked_loss
                kl_loss = None

            # For ZINB, "reconstruction" to store/plot is the mean parameter mu if pi is not too high otherwise set to 0
            recon_batch = mu * (pi < threshold).float()

            # Add regularization if specified
            reg_penalty = 0.0
            if regularizer.lower() == 'l1' and alpha > 0:
                parameters = list(model.parameters())
                if chrom and chrom_embedding is not None:
                    parameters += list(chrom_embedding.parameters())
                l1_penalty = l1_regularization(parameters)
                reg_penalty = alpha * l1_penalty.item()
                loss = loss + alpha * l1_penalty
            elif regularizer.lower() == 'l2' and alpha > 0:
                parameters = list(model.parameters())
                if chrom and chrom_embedding is not None:
                    parameters += list(chrom_embedding.parameters())
                l2_penalty = sum(torch.sum(p ** 2) for p in parameters)
                reg_penalty = alpha * l2_penalty.item()
                # Note: L2 is typically applied via weight_decay in optimizer, but we compute it here for tracking

            # -------- bookkeeping (single place) --------
            total_loss += loss.item() * batch_size

            if recon_loss is not None:
                total_recon_loss += recon_loss.item() * batch_size
            if kl_loss is not None:
                total_kl_loss += kl_loss.item() * batch_size
            if masked_loss is not None:
                total_masked_loss += masked_loss.item() * batch_size
            total_reg_loss += reg_penalty * batch_size

            # Store common outputs
            all_reconstructions.append(recon_batch.detach().cpu().numpy())
            all_latents.append(z.detach().cpu().numpy())
            all_originals.append(y.detach().cpu().numpy())

            # Store ZINB-specific outputs
            all_theta.append(theta.detach().cpu().numpy())
            all_pi.append(pi.detach().cpu().numpy())
            all_raw_counts.append(y_raw.detach().cpu().numpy())
            all_mu_raw.append(mu.detach().cpu().numpy())  # Store raw mu (unmasked)
            
            # Store mask from dataloader
            if denoise_percent > 0:
                all_masks.append(mask.detach().cpu().numpy())
    # ... after loop
    all_reconstructions = np.concatenate(all_reconstructions, axis=0)
    all_latents = np.concatenate(all_latents, axis=0)
    all_originals = np.concatenate(all_originals, axis=0)
    
    # Concatenate ZINB parameters
    if len(all_theta) > 0:
        all_theta = np.concatenate(all_theta, axis=0)
        all_pi = np.concatenate(all_pi, axis=0)
        all_raw_counts = np.concatenate(all_raw_counts, axis=0)
        all_mu_raw = np.concatenate(all_mu_raw, axis=0)
    else:
        all_theta = None
        all_pi = None
        all_raw_counts = None
        all_mu_raw = None
    
    # Concatenate masks from dataloader
    if denoise_percent > 0 and len(all_masks) > 0:
        all_masks = np.concatenate(all_masks, axis=0)
    else:
        all_masks = None
    
    # Calculate metrics
    test_loss = total_loss / len(all_originals)
    
    # For ZINB models, compare raw counts to predictions
    mae = mean_absolute_error(all_raw_counts.flatten(), all_reconstructions.flatten())
    r2 = r2_score(all_raw_counts.flatten(), all_reconstructions.flatten())
    
    # Build metrics dictionary
    test_recon_loss = total_recon_loss / len(all_originals)
    metrics = {'total_loss': test_loss, 'zinb_nll': test_recon_loss, 'mae': mae, 'r2': r2}
    
    if is_zinbvae:
        test_kl_loss = total_kl_loss / len(all_originals)
        metrics['kl_loss'] = test_kl_loss
    
    # Add masked loss if computed
    if gamma > 0 and denoise_percent > 0:
        test_masked_loss = total_masked_loss / len(all_originals)
        metrics['masked_loss'] = test_masked_loss
    
    # Add regularization loss if computed
    if regularizer.lower() != 'none' and alpha > 0:
        test_reg_loss = total_reg_loss / len(all_originals)
        metrics['reg_loss'] = test_reg_loss
    
    # Print metrics
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"Total Loss: {test_loss:.6f}")
    print(f"  - ZINB NLL: {test_recon_loss:.6f}")
    # print the number of masked values with a pi higher than the threshold and lower than the threshold
    print(f"Total number of masked values: {all_masks.sum()}")
    print(f"  - Number of values with pi >= {threshold}: {(all_pi[all_masks] >= threshold).sum()}")
    print(f"  - Number of values with pi < {threshold}: {(all_pi[all_masks] < threshold).sum()}")
    
    if is_zinbvae:
        print(f"  - KL Divergence: {test_kl_loss:.6f}")
    
    if gamma > 0 and denoise_percent > 0:
        print(f"  - Masked Reconstruction: {test_masked_loss:.6f}")
    
    if regularizer.lower() != 'none' and alpha > 0:
        print(f"  - Regularization: {test_reg_loss:.6f}")
    
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"R² Score: {r2:.6f}")
    print("="*50 + "\n")
    
    if plot:
        # Use ZINB plotting function
        model_type_str = model.model_type
        use_conv = model.use_conv if hasattr(model, 'use_conv') else False
        # Pass all_mu_raw (unmasked mu) for plotting pi vs mu relationship
        plot_zinb_test_results(all_originals, all_mu_raw, 
                              all_theta=all_theta, all_pi=all_pi, all_raw_counts=all_raw_counts,
                              all_masks=all_masks, denoise_percent=denoise_percent,
                              model_type=model_type_str, n_examples=n_examples, 
                              metrics=metrics, use_conv=use_conv, name=name, subdir=eval_mode)
        
    return all_reconstructions, all_latents, metrics



def parser_args():
    parser = argparse.ArgumentParser(description='Train and test ZINB Autoencoder models (ZINBAE, ZINBVAE)')
    parser.add_argument('--model', type=str, choices=['ZINBAE', 'ZINBVAE', 'both'], default='both',
                        help='Model type to train: ZINBAE, ZINBVAE, or both (default: both)')
    parser.add_argument('--use_conv', action='store_true',
                        help='Whether to use Conv1D layer in the model')
    parser.add_argument('--filename', type=str, default='',
                        help='Base filename for loading data (default: empty string)')
    parser.add_argument('--results_subdir', type=str, default='',
                        help='Subdirectory name for organizing results (e.g., "small_data"). Creates AE/results/training/<subdir>/ and AE/results/testing/<subdir>/')
    parser.add_argument('--denoise_percent', type=float, default=0,
                        help='Percentage of non-zero values to randomly set to zero for denoising (0.0 to 1.0, default: 0.3)')
    parser.add_argument('--sample_fraction', type=float, default=0.5,
                        help='Fraction of data to randomly sample for training (0.0 to 1.0, default: 0.5)')
    parser.add_argument('--no_test', action='store_true',
                        help='Evaluate on training data instead of test data')
    parser.add_argument('--chrom', action='store_true', help='Whether to use chromosome embedding')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs (default: 10)')
    parser.add_argument('--beta', type=float, default=1.0, help='Beta weight for KL divergence in VAE/ZINBVAE (default: 1.0)')
    parser.add_argument('--regularizer', type=str, choices=['none', 'L1', 'L2'], default='none', help='Regularization type: none, L1, or L2 (default: none)')
    parser.add_argument('--alpha', type=float, default=1e-4, help='Regularization strength (default: 0.0)')    
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability for regularization (0.0 to 1.0, default: 0.0)')
    parser.add_argument('--gamma', type=float, default=1.0, help='Weight for masked reconstruction loss in ZINB models (default: 0.0)')
    parser.add_argument('--pi_threshold', type=float, default=0.5, help='Threshold for zero-inflation probability in ZINB models (default: 0.5)')
    return parser.parse_args()

    
if __name__ == "__main__":
    args = parser_args()
    
    input_path = "Data/processed_data/"
    
    filename = args.filename
    results_subdir = args.results_subdir if args.results_subdir else ""
    no_test = args.no_test if args.no_test else False 
    chrom = args.chrom if args.chrom else False
    
    # Load data
    train_input_path = input_path + filename + "train_data.npy"
    print("Loading training data from:", train_input_path)
    train_dataloader = dataloader_from_array(train_input_path, chrom=chrom, batch_size=64, shuffle=True, zinb=True, sample_fraction=args.sample_fraction, denoise_percentage=args.denoise_percent)
    
    # If no_test, use training data for evaluation; otherwise load test data
    if no_test:
        print("Using training data for evaluation (--no_test flag set)")
        test_dataloader = train_dataloader
    else:
        test_input_path = input_path + filename + "test_data.npy"
        print("Loading test data from:", test_input_path)
        test_dataloader = dataloader_from_array(test_input_path, chrom=chrom, batch_size=64, shuffle=True, zinb=True, denoise_percentage=args.denoise_percent)
    
    # Print size of train data
    num_train_samples = len(train_dataloader.dataset)
    print(f"Number of training samples: {num_train_samples}")
    
    # Create chromosome embedding once to use consistently
    if chrom:
        chrom_embedding = ChromosomeEmbedding()
    else:
        chrom_embedding = None
        
    feature_dim = train_dataloader.dataset.tensors[0].shape[2] 
    feature_dim += 1  # +1 for y_in (the noisy input)
    feature_dim += chrom_embedding.embedding.embedding_dim if chrom else 0  # +4 for chromosome embedding
    
    # Train and test based on model choice
    if args.model in ['ZINBAE', 'both']:
        print("="*60)
        print("TRAINING ZINB AUTOENCODER (ZINBAE)")
        print("="*60)
        zinbae_model = ZINBAE(seq_length=2000, feature_dim=feature_dim, layers=[512, 256, 128], use_conv=args.use_conv, dropout=args.dropout)
        trained_zinbae = train(zinbae_model, train_dataloader, num_epochs=args.epochs, learning_rate=1e-3, 
                              chrom=chrom, chrom_embedding=chrom_embedding, plot=True, name=results_subdir, denoise_percent=args.denoise_percent,
                              regularizer=args.regularizer, alpha=args.alpha, gamma=args.gamma, pi_threshold=args.pi_threshold)
        
        zinbae_reconstructions, zinbae_latents, zinbae_metrics = test(trained_zinbae, test_dataloader, 
                                                                      chrom=chrom, chrom_embedding=chrom_embedding, 
                                                                      plot=True, n_examples=5, name=results_subdir, denoise_percent=args.denoise_percent,
                                                                      threshold=args.pi_threshold, gamma=args.gamma, pi_threshold=args.pi_threshold,
                                                                      regularizer=args.regularizer, alpha=args.alpha)
    
    if args.model in ['ZINBVAE', 'both']:
        if args.model == 'both':
            print("\n" + "="*60)
        else:
            print("="*60)
        print("TRAINING ZINB VARIATIONAL AUTOENCODER (ZINBVAE)")
        print("="*60)
        zinbvae_model = ZINBVAE(seq_length=2000, feature_dim=feature_dim, layers=[512, 256, 128], use_conv=args.use_conv, dropout=args.dropout)
        trained_zinbvae = train(zinbvae_model, train_dataloader, num_epochs=args.epochs, learning_rate=1e-3, 
                               chrom=chrom, chrom_embedding=chrom_embedding, plot=True, beta=args.beta, name=results_subdir, denoise_percent=args.denoise_percent,
                               regularizer=args.regularizer, alpha=args.alpha, gamma=args.gamma, pi_threshold=args.pi_threshold)
        
        zinbvae_reconstructions, zinbvae_latents, zinbvae_metrics = test(trained_zinbvae, test_dataloader, 
                                                                         chrom=chrom, chrom_embedding=chrom_embedding, 
                                                                         plot=True, n_examples=5, beta=args.beta, name=results_subdir, denoise_percent=args.denoise_percent,
                                                                         threshold=args.pi_threshold, gamma=args.gamma, pi_threshold=args.pi_threshold,
                                                                         regularizer=args.regularizer, alpha=args.alpha)