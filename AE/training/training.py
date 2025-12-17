import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, r2_score
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))) 
from AE.training.results import plot_binary_training_loss, plot_test_results, plot_training_loss, plot_binary_test_results, plot_zinb_training_loss, plot_zinb_test_results
import argparse
from AE.architectures.Autoencoder import AE, VAE
from AE.architectures.Autoencoder_binary import AE_binary, VAE_binary
from AE.architectures.ZINBAE import ZINBAE, ZINBVAE
from AE.training.loss_functions import zinb_nll
from AE.training.training_utils import ChromosomeEmbedding, add_noise, dataloader_from_array, gaussian_kl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def train(model, dataloader, num_epochs=50, learning_rate=1e-3, chrom=True, chrom_embedding=None, plot=True, beta=1.0, binary = False, name="", denoise_percent=0.3):
    """
    Train AE, VAE, ZINBAE, or ZINBVAE model
    
    Parameters:
    -----------
    model : AE, VAE, ZINBAE, or ZINBVAE
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
        Weight for KL divergence loss (only used for VAE/ZINBVAE). Default=1.0
    binary : bool
        Whether to use binary AE/VAE models. Default=False
    denoise_percent : float
        Percentage of non-zero values to randomly set to zero for denoising (0.0 to 1.0). Default=0.0
    """
    model.to(device)
    parameters = list(model.parameters())
    if chrom:
        if chrom_embedding is None:
            raise ValueError("chrom_embedding must be provided when chrom=True")
        chrom_embedding.to(device)
        parameters += list(chrom_embedding.parameters())

    # Determine model type
    is_zinb = getattr(model, "model_type", None) in {"ZINBAE", "ZINBVAE"}
    is_vae  = getattr(model, "model_type", None) in {"VAE", "VAE_binary", "ZINBVAE"}
    
    if is_zinb and binary:
        raise ValueError("binary=True is not supported for ZINB models in this setup.")
    
    # Set criterion (not used for ZINB models)
    if not is_zinb:
        if binary:
            criterion = nn.BCELoss(reduction='mean')  # Mean reduction per batch
        else:
            criterion = nn.MSELoss(reduction='mean')  # Mean reduction per batch
    
    optimizer = optim.Adam(parameters, lr=learning_rate)
    
    epoch_losses = []
    epoch_recon_losses = []  # For ZINBVAE/VAE
    epoch_kl_losses = []      # For ZINBVAE/VAE
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            # Unpack batch based on flags
            if is_zinb:
                if chrom:
                    x, y, c, y_raw, size_factors = batch
                    c = c.to(device)
                else:
                    x, y, y_raw, size_factors = batch
                y_raw = y_raw.to(device)  # (B, seq) - raw counts
                size_factors = size_factors.to(device)  # (B,) - size factors
            else:
                if chrom:
                    if binary:
                        x, y, y_binary, c = batch
                    else:
                        x, y, c = batch
                    c = c.to(device)
                else:
                    if binary:
                        x, y, y_binary = batch
                    else:
                        x, y = batch
            
            x = x.to(device)         # (B, seq, F_other or F_other_without_chr)
            y = y.to(device)         # (B, seq) - normalized counts
            if binary:
                y_binary = y_binary.to(device)  # (B, seq)
            
            optimizer.zero_grad()
            
            # Apply denoising: randomly set some non-zero values to zero in the input
            y_noisy, mask = add_noise(y, denoise_percent)
            y_in = y_noisy.unsqueeze(-1)  # Add feature dimension
            if chrom:
                c_emb = chrom_embedding(c)
                batch_input = torch.cat((y_in, x, c_emb), dim=2)
            else:
                batch_input = torch.cat((y_in, x), dim=2)

            # reset per batch
            recon_loss = None
            kl_loss = None

            # Forward pass
            if is_zinb:
                if model.model_type == 'ZINBVAE':
                    mu, theta, pi, z, mu_z, logvar_z = model(batch_input, size_factors)
                    recon_loss = zinb_nll(y_raw, mu, theta, pi, reduction='mean')
                    kl_loss = gaussian_kl(mu_z, logvar_z)
                    loss = recon_loss + beta * kl_loss
                else:  # ZINBAE
                    mu, theta, pi, z = model(batch_input, size_factors)
                    recon_loss = zinb_nll(y_raw, mu, theta, pi, reduction='mean')
                    loss = recon_loss

            elif is_vae:
                recon_batch, z, mu, logvar = model(batch_input)
                recon_loss = criterion(recon_batch, y_binary if binary else y)
                kl_loss = gaussian_kl(mu, logvar)
                loss = recon_loss + beta * kl_loss

            else:  # AE
                recon_batch, z = model(batch_input)
                recon_loss = criterion(recon_batch, y_binary if binary else y)
                loss = recon_loss

            # bookkeeping
            epoch_recon_loss += recon_loss.item() * y.size(0)
            if kl_loss is not None:
                epoch_kl_loss += kl_loss.item() * y.size(0)
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients (especially important for VAE/ZINBVAE)
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
            
            optimizer.step()
            
            # Check for NaN in loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nWARNING: NaN or Inf detected in loss at epoch {epoch+1}!")
                print(f"Batch info: batch size={y.size(0)}")
                if is_zinb:
                    print(f"  mu: min={mu.min():.4f}, max={mu.max():.4f}, mean={mu.mean():.4f}")
                    print(f"  theta: min={theta.min():.4f}, max={theta.max():.4f}, mean={theta.mean():.4f}")
                    print(f"  pi: min={pi.min():.4f}, max={pi.max():.4f}, mean={pi.mean():.4f}")
                    if model.model_type == 'ZINBVAE':
                        print(f"  mu_z: min={mu_z.min():.4f}, max={mu_z.max():.4f}, mean={mu_z.mean():.4f}")
                        print(f"  logvar_z: min={logvar_z.min():.4f}, max={logvar_z.max():.4f}, mean={logvar_z.mean():.4f}")
                # Skip this batch or break
                continue
            
            epoch_loss += loss.item() * y.size(0)
            
            # Update progress bar with current loss
            if is_vae or (is_zinb and model.model_type == 'ZINBVAE'):
                pbar.set_postfix({'total_loss': f'{loss.item():.4f}', 
                                 'recon': f'{recon_loss.item():.4f}',
                                 'kl': f'{kl_loss.item():.4f}'})
            else:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        epoch_loss /= len(dataloader.dataset)
        epoch_losses.append(epoch_loss)
        
        if is_vae:
            epoch_recon_loss /= len(dataloader.dataset)
            epoch_kl_loss /= len(dataloader.dataset)
            epoch_recon_losses.append(epoch_recon_loss)
            epoch_kl_losses.append(epoch_kl_loss)
            print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {epoch_loss:.4f}, "
                  f"Recon: {epoch_recon_loss:.4f}, KL: {epoch_kl_loss:.4f}")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    if plot:
        if binary:
            model_type_str = model.model_type if hasattr(model, 'model_type') else 'AE_binary'
            use_conv = model.use_conv if hasattr(model, 'use_conv') else False
            plot_binary_training_loss(epoch_losses, model_type=model_type_str, use_conv=use_conv, name=name)
        elif is_zinb:
            model_type_str = model.model_type
            use_conv = model.use_conv if hasattr(model, 'use_conv') else False
            if model.model_type == 'ZINBVAE':
                plot_zinb_training_loss(epoch_losses, epoch_recon_losses, epoch_kl_losses, 
                                       model_type=model_type_str, use_conv=use_conv, name=name)
            else:
                plot_zinb_training_loss(epoch_losses, model_type=model_type_str, use_conv=use_conv, name=name)
        else:
            model_type_str = model.model_type if hasattr(model, 'model_type') else 'AE'
            use_conv = model.use_conv if hasattr(model, 'use_conv') else False
            plot_training_loss(epoch_losses, model_type=model_type_str, use_conv=use_conv, name=name)
    
    return model

def test(model, dataloader, chrom=True, chrom_embedding=None, plot=True, n_examples=5, beta=1.0, binary=False, name="", denoise_percent=0.0):
    """
    Test AE, VAE, ZINBAE, or ZINBVAE model
    
    Parameters:
    -----------
    model : AE, VAE, ZINBAE, or ZINBVAE
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
        Weight for KL divergence loss (only used for VAE/ZINBVAE). Default=1.0
    binary : bool
        Whether to use binary AE/VAE models. Default=False
    denoise_percent : float
        Percentage of non-zero values to randomly set to zero for denoising (0.0 to 1.0). Default=0.0
    """
    model.to(device)
    model.eval()
    
    if chrom:
        if chrom_embedding is None:
            raise ValueError("chrom_embedding must be provided when chrom=True")
        chrom_embedding.to(device)
        chrom_embedding.eval()
    
    all_reconstructions = []
    all_latents = []
    all_originals = []
    all_theta = []  # For ZINB models
    all_pi = []     # For ZINB models
    all_raw_counts = []  # For ZINB models
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    
    # Determine model type
    is_zinb = getattr(model, "model_type", None) in {"ZINBAE", "ZINBVAE"}
    is_vae = getattr(model, "model_type", None) in {"VAE", "VAE_binary", "ZINBVAE"}
    
    # Set criterion (not used for ZINB models)
    if not is_zinb:
        if binary:
            criterion = nn.BCELoss(reduction='mean')  # Mean reduction per batch
        else:
            criterion = nn.MSELoss(reduction='mean')  # Mean reduction per batch
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            # Unpack batch based on flags
            if is_zinb:
                if chrom:
                    x, y, c, y_raw, size_factors = batch
                    c = c.to(device)
                else:
                    x, y, y_raw, size_factors = batch
                y_raw = y_raw.to(device)  # (B, seq) - raw counts
                size_factors = size_factors.to(device)  # (B,) - size factors
            else:
                if chrom:
                    if binary:
                        x, y, y_binary, c = batch
                    else:
                        x, y, c = batch
                    c = c.to(device)
                else:
                    if binary:
                        x, y, y_binary = batch
                    else:
                        x, y = batch
            
            x = x.to(device)         # (B, seq, F_other or F_other_without_chr)
            y = y.to(device)         # (B, seq) - normalized counts
            if binary:
                y_binary = y_binary.to(device)  # (B, seq)
            
            # Apply denoising: randomly set some non-zero values to zero in the input
            y_noisy, mask = add_noise(y, denoise_percent)
            y_in = y_noisy.unsqueeze(-1)  # Add feature dimension
            if chrom:
                c_emb = chrom_embedding(c)
                batch_input = torch.cat((y_in, x, c_emb), dim=2)
            else:
                batch_input = torch.cat((y_in, x), dim=2)


            target = y_binary if binary else y
            batch_size = y.size(0)

            recon_loss = None
            kl_loss = None
            theta = pi = None  # only used for ZINB

            # -------- forward + losses --------
            if is_zinb:
                out = model(batch_input, size_factors)

                if model.model_type == "ZINBVAE":
                    mu, theta, pi, z, mu_z, logvar_z = out
                    recon_loss = zinb_nll(y_raw, mu, theta, pi, reduction="mean")
                    kl_loss = gaussian_kl(mu_z, logvar_z)
                    loss = recon_loss + beta * kl_loss
                else:  # ZINBAE
                    mu, theta, pi, z = out
                    recon_loss = zinb_nll(y_raw, mu, theta, pi, reduction="mean")
                    loss = recon_loss

                # For ZINB, "reconstruction" to store/plot is the mean parameter mu
                recon_batch = mu

            elif is_vae:
                recon_batch, z, mu, logvar = model(batch_input)
                recon_loss = criterion(recon_batch, target)
                kl_loss = gaussian_kl(mu, logvar)
                loss = recon_loss + beta * kl_loss

            else:  # AE
                recon_batch, z = model(batch_input)
                recon_loss = criterion(recon_batch, target)
                loss = recon_loss

            # -------- bookkeeping (single place) --------
            total_loss += loss.item() * batch_size

            if recon_loss is not None:
                total_recon_loss += recon_loss.item() * batch_size
            if kl_loss is not None:
                total_kl_loss += kl_loss.item() * batch_size

            # Store common outputs
            all_reconstructions.append(recon_batch.detach().cpu().numpy())
            all_latents.append(z.detach().cpu().numpy())
            all_originals.append(y.detach().cpu().numpy())

            # Store ZINB-specific outputs
            if is_zinb:
                all_theta.append(theta.detach().cpu().numpy())
                all_pi.append(pi.detach().cpu().numpy())
                all_raw_counts.append(y_raw.detach().cpu().numpy())

    # ... after loop
    all_reconstructions = np.concatenate(all_reconstructions, axis=0)
    all_latents = np.concatenate(all_latents, axis=0)
    all_originals = np.concatenate(all_originals, axis=0)
    
    # Concatenate ZINB parameters if collected
    if is_zinb and len(all_theta) > 0:
        all_theta = np.concatenate(all_theta, axis=0)
        all_pi = np.concatenate(all_pi, axis=0)
        all_raw_counts = np.concatenate(all_raw_counts, axis=0)
    else:
        all_theta = None
        all_pi = None
        all_raw_counts = None
    
    # Calculate metrics
    test_loss = total_loss / len(all_originals)
    mae = mean_absolute_error(all_originals.flatten(), all_reconstructions.flatten())
    r2 = r2_score(all_originals.flatten(), all_reconstructions.flatten())
    
    # Build metrics dictionary
    if is_zinb:
        metrics = {'zinb_nll': test_loss, 'mae': mae, 'r2': r2}
    else:
        metrics = {'mse': test_loss, 'mae': mae, 'r2': r2}
    
    if is_vae or (is_zinb and model.model_type == 'ZINBVAE'):
        test_recon_loss = total_recon_loss / len(all_originals)
        test_kl_loss = total_kl_loss / len(all_originals)
        metrics['recon_loss'] = test_recon_loss
        metrics['kl_loss'] = test_kl_loss
    
    # Print metrics
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    if is_zinb:
        print(f"Test Loss (ZINB NLL): {test_loss:.6f}")
    else:
        print(f"Test Loss (MSE): {test_loss:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"R² Score: {r2:.6f}")
    
    if is_vae or (is_zinb and model.model_type == 'ZINBVAE'):
        print(f"Reconstruction Loss: {test_recon_loss:.6f}")
        print(f"KL Divergence: {test_kl_loss:.6f}")
    
    print("="*50 + "\n")
    
    if plot:
        if binary:
            model_type_str = model.model_type if hasattr(model, 'model_type') else 'AE_binary'
            use_conv = model.use_conv if hasattr(model, 'use_conv') else False
            plot_binary_test_results(all_originals, all_reconstructions, all_reconstructions,
                                    model_type=model_type_str, n_examples=n_examples, metrics=metrics, use_conv=use_conv, name=name )
        elif is_zinb:
            # Use special ZINB plotting function
            model_type_str = model.model_type
            use_conv = model.use_conv if hasattr(model, 'use_conv') else False
            plot_zinb_test_results(all_originals, all_reconstructions, 
                                  all_theta=all_theta, all_pi=all_pi, all_raw_counts=all_raw_counts,
                                  model_type=model_type_str, n_examples=n_examples, 
                                  metrics=metrics, use_conv=use_conv, name=name)
        else:
            model_type_str = model.model_type if hasattr(model, 'model_type') else 'AE'
            use_conv = model.use_conv if hasattr(model, 'use_conv') else False
            plot_test_results(all_originals, all_reconstructions, model_type=model_type_str, 
                            n_examples=n_examples, metrics=metrics, use_conv=use_conv, name=name)
        
    return all_reconstructions, all_latents, metrics



def parser_args():
    parser = argparse.ArgumentParser(description='Train and test Autoencoder models (AE, VAE, ZINBAE, ZINBVAE)')
    parser.add_argument('--model', type=str, choices=['AE', 'VAE', 'ZINBAE', 'ZINBVAE', 'both', 'all'], default='both',
                        help='Model type to train: AE, VAE, ZINBAE, ZINBVAE, both (AE+VAE), or all (default: both)')
    parser.add_argument('--use_conv', action='store_true',
                        help='Whether to use Conv1D layer in the model')
    parser.add_argument('--binary', action='store_true',
                        help='Whether to use binary AE/VAE models')
    parser.add_argument('--filename', type=str, default='',
                        help='Base filename for loading data (default: empty string)')
    parser.add_argument('--denoise_percent', type=float, default=0,
                        help='Percentage of non-zero values to randomly set to zero for denoising (0.0 to 1.0, default: 0.3)')
    parser.add_argument('--sample_fraction', type=float, default=0.25,
                        help='Fraction of data to randomly sample for training (0.0 to 1.0, default: 0.25)')
    return parser.parse_args()

    
if __name__ == "__main__":
    args = parser_args()
    
    input_path = "Data/processed_data/"
    
    filename = args.filename
    
    # Determine if we're using ZINB models
    is_zinb = args.model in ['ZINBAE', 'ZINBVAE', 'all']
    
    # Load data
    train_input_path = input_path + filename + "train_data.npy"
    print("Loading training data from:", train_input_path)
    train_dataloader = dataloader_from_array(train_input_path, chrom=True, batch_size=64, shuffle=True, binary=args.binary, zinb=is_zinb, sample_fraction=args.sample_fraction)
    
    test_input_path = input_path + filename + "test_data.npy"
    print("Loading test data from:", test_input_path)
    test_dataloader = dataloader_from_array(test_input_path, chrom=True, batch_size=64, shuffle=False, binary=args.binary, zinb=is_zinb)
    
    
    # Create chromosome embedding once to use consistently
    chrom = True 
    if chrom:
        chrom_embedding = ChromosomeEmbedding()
    else:
        chrom_embedding = None
    
    
    # Train and test based on model choice
    if args.model in ['AE', 'both', 'all']:
        print("="*60)
        print("TRAINING AUTOENCODER (AE)")
        print("="*60)
        if args.binary:
            ae_model = AE_binary(seq_length=2000, feature_dim=8, layers=[512, 256, 128], use_conv=args.use_conv)
        else:
            ae_model = AE(seq_length=2000, feature_dim=8, layers=[512, 256, 128], use_conv=args.use_conv)
        trained_ae = train(ae_model, train_dataloader, num_epochs=10, learning_rate=1e-3, 
                          chrom=chrom, chrom_embedding=chrom_embedding, plot=True, binary=args.binary, name=filename)
        ae_reconstructions, ae_latents, ae_metrics = test(trained_ae, test_dataloader, 
                                                          chrom=True, chrom_embedding=chrom_embedding, 
                                                          plot=True, n_examples=5, binary=args.binary, name=filename)
    
    if args.model in ['VAE', 'both', 'all']:
        if args.model in ['both', 'all']:
            print("\n" + "="*60)
        else:
            print("="*60)
        print("TRAINING VARIATIONAL AUTOENCODER (VAE)")
        print("="*60)
        if args.binary:
            vae_model = VAE_binary(seq_length=2000, feature_dim=8, layers=[512, 256, 128], use_conv=args.use_conv)
        else:   
            vae_model = VAE(seq_length=2000, feature_dim=8, layers=[512, 256, 128], use_conv=args.use_conv)
        trained_vae = train(vae_model, train_dataloader, num_epochs=10, learning_rate=1e-3, 
                           chrom=chrom, chrom_embedding=chrom_embedding, plot=True, beta=1.0, binary=args.binary, name=filename, denoise_percent=0)
        vae_reconstructions, vae_latents, vae_metrics = test(trained_vae, test_dataloader, 
                                                             chrom=True, chrom_embedding=chrom_embedding, 
                                                             plot=True, n_examples=5, beta=1.0, binary=args.binary, name=filename, denoise_percent=0)
    
    if args.model in ['ZINBAE', 'all']:
        if args.model == 'all':
            print("\n" + "="*60)
        else:
            print("="*60)
        print("TRAINING ZINB AUTOENCODER (ZINBAE)")
        print("="*60)
        zinbae_model = ZINBAE(seq_length=2000, feature_dim=8, layers=[512, 256, 128], use_conv=args.use_conv)
        trained_zinbae = train(zinbae_model, train_dataloader, num_epochs=10, learning_rate=1e-3, 
                              chrom=chrom, chrom_embedding=chrom_embedding, plot=True, name=filename, denoise_percent=0.3)
        zinbae_reconstructions, zinbae_latents, zinbae_metrics = test(trained_zinbae, test_dataloader, 
                                                                      chrom=True, chrom_embedding=chrom_embedding, 
                                                                      plot=True, n_examples=5, name=filename, denoise_percent=0)
    
    if args.model in ['ZINBVAE', 'all']:
        if args.model == 'all':
            print("\n" + "="*60)
        else:
            print("="*60)
        print("TRAINING ZINB VARIATIONAL AUTOENCODER (ZINBVAE)")
        print("="*60)
        zinbvae_model = ZINBVAE(seq_length=2000, feature_dim=8, layers=[512, 256, 128], use_conv=args.use_conv)
        trained_zinbvae = train(zinbvae_model, train_dataloader, num_epochs=10, learning_rate=1e-3, 
                               chrom=chrom, chrom_embedding=chrom_embedding, plot=True, beta=1.0, name=filename, denoise_percent=args.denoise_percent)
        zinbvae_reconstructions, zinbvae_latents, zinbvae_metrics = test(trained_zinbvae, test_dataloader, 
                                                                         chrom=True, chrom_embedding=chrom_embedding, 
                                                                         plot=True, n_examples=5, beta=1.0, name=filename, denoise_percent=args.denoise_percent)