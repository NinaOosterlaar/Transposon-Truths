import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, r2_score
from results import plot_binary_training_loss, plot_test_results, plot_training_loss, plot_binary_test_results
import argparse
from Autoencoder import AE, VAE
from Autoencoder_binary import AE_binary, VAE_binary
from ZINBAE import ZINBAE, ZINBVAE
from loss_functions import zinb_nll

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def train(model, dataloader, num_epochs=50, learning_rate=1e-3, chrom=True, chrom_embedding=None, plot=True, beta=1.0, binary = False, name=""):
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
    """
    model.to(device)
    parameters = list(model.parameters())
    if chrom:
        if chrom_embedding is None:
            raise ValueError("chrom_embedding must be provided when chrom=True")
        chrom_embedding.to(device)
        parameters += list(chrom_embedding.parameters())

    # Determine model type
    is_zinb = hasattr(model, 'model_type') and (model.model_type == 'ZINBAE' or model.model_type == 'ZINBVAE')
    is_vae = hasattr(model, 'model_type') and (model.model_type == 'VAE' or model.model_type == 'VAE_binary' or model.model_type == 'ZINBVAE')
    
    # Set criterion (not used for ZINB models)
    if not is_zinb:
        if binary:
            criterion = nn.BCELoss(reduction='sum')  # Use sum reduction for proper scaling with KL loss
        else:
            criterion = nn.MSELoss(reduction='sum')  # Use sum reduction for proper scaling with KL loss
    
    optimizer = optim.Adam(parameters, lr=learning_rate)
    
    epoch_losses = []
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
            y_in = y.unsqueeze(-1)  # Add feature dimension
            if chrom:
                c_emb = chrom_embedding(c)
                batch_input = torch.cat((y_in, x, c_emb), dim=2)
            else:
                batch_input = torch.cat((y_in, x), dim=2)
            
            # Forward pass
            if is_zinb:
                # ZINB models
                if model.model_type == 'ZINBVAE':
                    mu, theta, pi, z, mu_z, logvar_z = model(batch_input, size_factors)
                    # ZINB reconstruction loss (sum over batch and seq, then average over batch)
                    recon_loss = zinb_nll(y_raw, mu, theta, pi, reduction='sum') / y.size(0)
                    # KL divergence for latent space
                    kl_loss = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp()) / y.size(0)
                    loss = recon_loss + beta * kl_loss
                    
                    epoch_recon_loss += recon_loss.item() * y.size(0)
                    epoch_kl_loss += kl_loss.item() * y.size(0)
                else:  # ZINBAE
                    mu, theta, pi, z = model(batch_input, size_factors)
                    # ZINB reconstruction loss (sum over batch and seq, then average over batch)
                    loss = zinb_nll(y_raw, mu, theta, pi, reduction='sum') / y.size(0)
            elif is_vae:
                # Regular VAE models
                recon_batch, z, mu, logvar = model(batch_input)
                # VAE loss = Reconstruction loss + KL divergence
                recon_loss = criterion(recon_batch, y_binary if binary else y) / y.size(0)  # Divide by batch size
                # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / y.size(0)
                loss = recon_loss + beta * kl_loss
                
                epoch_recon_loss += recon_loss.item() * y.size(0)
                epoch_kl_loss += kl_loss.item() * y.size(0)
            else:
                # Regular AE models
                recon_batch, z = model(batch_input)
                loss = criterion(recon_batch, y_binary if binary else y) / y.size(0)  # Divide by batch size for consistency
            
            loss.backward()
            optimizer.step()
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
        
        if is_vae or (is_zinb and model.model_type == 'ZINBVAE'):
            epoch_recon_loss /= len(dataloader.dataset)
            epoch_kl_loss /= len(dataloader.dataset)
            print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {epoch_loss:.4f}, "
                  f"Recon: {epoch_recon_loss:.4f}, KL: {epoch_kl_loss:.4f}")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    if plot:
        if binary:
            model_type_str = model.model_type if hasattr(model, 'model_type') else 'AE_binary'
            use_conv = model.use_conv if hasattr(model, 'use_conv') else False
            plot_binary_training_loss(epoch_losses, model_type=model_type_str, use_conv=use_conv, name=name)
        else:
            model_type_str = model.model_type if hasattr(model, 'model_type') else 'AE'
            use_conv = model.use_conv if hasattr(model, 'use_conv') else False
            plot_training_loss(epoch_losses, model_type=model_type_str, use_conv=use_conv, name=name)
    
    return model

def test(model, dataloader, chrom=True, chrom_embedding=None, plot=True, n_examples=5, beta=1.0, binary=False, name=""):
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
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    
    # Determine model type
    is_zinb = hasattr(model, 'model_type') and (model.model_type == 'ZINBAE' or model.model_type == 'ZINBVAE')
    is_vae = hasattr(model, 'model_type') and (model.model_type == 'VAE' or model.model_type == 'VAE_binary' or model.model_type == 'ZINBVAE')
    
    # Set criterion (not used for ZINB models)
    if not is_zinb:
        if binary:
            criterion = nn.BCELoss(reduction='sum')  # Use sum reduction for proper scaling
        else:
            criterion = nn.MSELoss(reduction='sum')  # Use sum reduction for proper scaling
    
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
            
            y_in = y.unsqueeze(-1)  # Add feature dimension
            if chrom:
                c_emb = chrom_embedding(c)
                batch_input = torch.cat((y_in, x, c_emb), dim=2)
            else:
                batch_input = torch.cat((y_in, x), dim=2)
            
            # Forward pass
            if is_zinb:
                # ZINB models
                if model.model_type == 'ZINBVAE':
                    mu, theta, pi, z, mu_z, logvar_z = model(batch_input, size_factors)
                    # ZINB reconstruction loss (sum over batch and seq, then average over batch)
                    recon_loss = zinb_nll(y_raw, mu, theta, pi, reduction='sum') / y.size(0)
                    # KL divergence for latent space
                    kl_loss = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp()) / y.size(0)
                    loss = recon_loss + beta * kl_loss
                    
                    total_recon_loss += recon_loss.item() * y.size(0)
                    total_kl_loss += kl_loss.item() * y.size(0)
                    
                    # For ZINB, reconstruction is mu (mean of ZINB distribution)
                    recon_batch = mu
                else:  # ZINBAE
                    mu, theta, pi, z = model(batch_input, size_factors)
                    # ZINB reconstruction loss (sum over batch and seq, then average over batch)
                    loss = zinb_nll(y_raw, mu, theta, pi, reduction='sum') / y.size(0)
                    
                    # For ZINB, reconstruction is mu (mean of ZINB distribution)
                    recon_batch = mu
            elif is_vae:
                # Regular VAE models
                recon_batch, z, mu, logvar = model(batch_input)
                # Calculate losses
                recon_loss = criterion(recon_batch, y_binary if binary else y) / y.size(0)  # Divide by batch size
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / y.size(0)
                loss = recon_loss + beta * kl_loss
                
                total_recon_loss += recon_loss.item() * y.size(0)
                total_kl_loss += kl_loss.item() * y.size(0)
            else:
                # Regular AE models
                recon_batch, z = model(batch_input)
                loss = criterion(recon_batch, y_binary if binary else y) / y.size(0)  # Divide by batch size
            
            total_loss += loss.item() * y.size(0)
            
            all_reconstructions.append(recon_batch.cpu().numpy())
            all_latents.append(z.cpu().numpy())
            all_originals.append(y.cpu().numpy())
    
    all_reconstructions = np.concatenate(all_reconstructions, axis=0)
    all_latents = np.concatenate(all_latents, axis=0)
    all_originals = np.concatenate(all_originals, axis=0)
    
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
        else:
            model_type_str = model.model_type if hasattr(model, 'model_type') else 'AE'
            use_conv = model.use_conv if hasattr(model, 'use_conv') else False
            plot_test_results(all_originals, all_reconstructions, model_type=model_type_str, 
                            n_examples=n_examples, metrics=metrics, use_conv=use_conv, name=name)
        
    return all_reconstructions, all_latents, metrics

# Embed the chromosome feature if needed
class ChromosomeEmbedding(nn.Module):
    def __init__(self, num_chromosomes=17, embedding_dim=4):
        super(ChromosomeEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_chromosomes, embedding_dim)
        
    def forward(self, x):
        return self.embedding(x)

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
    train_dataloader = dataloader_from_array(train_input_path, chrom=True, batch_size=64, shuffle=True, binary=args.binary, zinb=is_zinb)
    
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
                           chrom=chrom, chrom_embedding=chrom_embedding, plot=True, beta=1.0, binary=args.binary, name=filename)
        vae_reconstructions, vae_latents, vae_metrics = test(trained_vae, test_dataloader, 
                                                             chrom=True, chrom_embedding=chrom_embedding, 
                                                             plot=True, n_examples=5, beta=1.0, binary=args.binary, name=filename)
    
    if args.model in ['ZINBAE', 'all']:
        if args.model == 'all':
            print("\n" + "="*60)
        else:
            print("="*60)
        print("TRAINING ZINB AUTOENCODER (ZINBAE)")
        print("="*60)
        zinbae_model = ZINBAE(seq_length=2000, feature_dim=8, layers=[512, 256, 128], use_conv=args.use_conv)
        trained_zinbae = train(zinbae_model, train_dataloader, num_epochs=10, learning_rate=1e-3, 
                              chrom=chrom, chrom_embedding=chrom_embedding, plot=True, name=filename)
        zinbae_reconstructions, zinbae_latents, zinbae_metrics = test(trained_zinbae, test_dataloader, 
                                                                      chrom=True, chrom_embedding=chrom_embedding, 
                                                                      plot=True, n_examples=5, name=filename)
    
    if args.model in ['ZINBVAE', 'all']:
        if args.model == 'all':
            print("\n" + "="*60)
        else:
            print("="*60)
        print("TRAINING ZINB VARIATIONAL AUTOENCODER (ZINBVAE)")
        print("="*60)
        zinbvae_model = ZINBVAE(seq_length=2000, feature_dim=8, layers=[512, 256, 128], use_conv=args.use_conv)
        trained_zinbvae = train(zinbvae_model, train_dataloader, num_epochs=10, learning_rate=1e-3, 
                               chrom=chrom, chrom_embedding=chrom_embedding, plot=True, beta=1.0, name=filename)
        zinbvae_reconstructions, zinbvae_latents, zinbvae_metrics = test(trained_zinbvae, test_dataloader, 
                                                                         chrom=True, chrom_embedding=chrom_embedding, 
                                                                         plot=True, n_examples=5, beta=1.0, name=filename)