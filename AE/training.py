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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def train(model, dataloader, num_epochs=50, learning_rate=1e-3, chrom=True, chrom_embedding=None, plot=True, beta=1.0, binary = False, name=""):
    """
    Train AE or VAE model
    
    Parameters:
    -----------
    model : AE or VAE
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
        Weight for KL divergence loss (only used for VAE). Default=1.0
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

    if binary:
        criterion = nn.BCELoss(reduction='sum')  # Use sum reduction for proper scaling with KL loss
    else:
        criterion = nn.MSELoss(reduction='sum')  # Use sum reduction for proper scaling with KL loss
    optimizer = optim.Adam(parameters, lr=learning_rate)
    
    is_vae = hasattr(model, 'model_type') and (model.model_type == 'VAE' or model.model_type == 'VAE_binary')
    
    epoch_losses = []
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            if chrom:
                if binary:
                    x, y, y_binary, c = batch      # dataloader has (x, y, c)
                else:
                    x, y, c = batch      # dataloader has (x, y, c)
                c = c.to(device)
            else:
                if binary:
                    x, y, y_binary = batch  # dataloader has (x, y)
                else:
                    x, y = batch  # dataloader has (x, y)
            
            x = x.to(device)         # (B, seq, F_other or F_other_without_chr)
            y = y.to(device)         # (B, seq)
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
            if is_vae:
                recon_batch, z, mu, logvar = model(batch_input)
                # VAE loss = Reconstruction loss + KL divergence
                recon_loss = criterion(recon_batch, y_binary if binary else y) / y.size(0)  # Divide by batch size
                # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / y.size(0)
                loss = recon_loss + beta * kl_loss
                
                epoch_recon_loss += recon_loss.item() * y.size(0)
                epoch_kl_loss += kl_loss.item() * y.size(0)
            else:
                recon_batch, z = model(batch_input)
                loss = criterion(recon_batch, y_binary if binary else y) / y.size(0)  # Divide by batch size for consistency
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * y.size(0)
            
            # Update progress bar with current loss
            if is_vae:
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
    Test AE or VAE model
    
    Parameters:
    -----------
    model : AE or VAE
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
        Weight for KL divergence loss (only used for VAE). Default=1.0
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
    if binary:
        criterion = nn.BCELoss(reduction='sum')  # Use sum reduction for proper scaling
    else:
        criterion = nn.MSELoss(reduction='sum')  # Use sum reduction for proper scaling
    
    is_vae = hasattr(model, 'model_type') and (model.model_type == 'VAE' or model.model_type == 'VAE_binary')
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            if chrom:
                if binary:
                    x, y, y_binary, c = batch      # dataloader has (x, y, c)
                else:
                    x, y, c = batch      # dataloader has (x, y, c)
                c = c.to(device)
            else:
                if binary:
                    x, y, y_binary = batch  # dataloader has (x, y)
                else:
                    x, y = batch  # dataloader has (x, y)
            
            x = x.to(device)         # (B, seq, F_other or F_other_without_chr)
            y = y.to(device)         # (B, seq)
            if binary:
                y_binary = y_binary.to(device)  # (B, seq)
            y_in = y.unsqueeze(-1)  # Add feature dimension
            if chrom:
                c_emb = chrom_embedding(c)
                batch_input = torch.cat((y_in, x, c_emb), dim=2)
            else:
                batch_input = torch.cat((y_in, x), dim=2)
            
            # Forward pass
            if is_vae:
                recon_batch, z, mu, logvar = model(batch_input)
                # Calculate losses
                recon_loss = criterion(recon_batch, y_binary if binary else y) / y.size(0)  # Divide by batch size
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / y.size(0)
                loss = recon_loss + beta * kl_loss
                
                total_recon_loss += recon_loss.item() * y.size(0)
                total_kl_loss += kl_loss.item() * y.size(0)
            else:
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
    metrics = {'mse': test_loss, 'mae': mae, 'r2': r2}
    if is_vae:
        test_recon_loss = total_recon_loss / len(all_originals)
        test_kl_loss = total_kl_loss / len(all_originals)
        metrics['recon_loss'] = test_recon_loss
        metrics['kl_loss'] = test_kl_loss
    
    # Print metrics
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"Test Loss (MSE): {test_loss:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"R² Score: {r2:.6f}")
    
    if is_vae:
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

def dataloader_from_array(input, chrom=True, batch_size=64, shuffle=True, binary=False):
    data_array = np.load(input)
    counts = data_array[:, :, 0]
    
    if binary:
        # Create binary targets based on counts > 0
        y_binary = (counts > 0).astype(np.float32)
    
    if chrom:
        features = data_array[:, :, 1:-1]
        chrom_indices = data_array[:, :, -1].astype(np.int64)
    else:
        features = data_array[:, :, 1:]
    
    x_tensor = torch.tensor(features, dtype=torch.float32)
    y_tensor = torch.tensor(counts, dtype=torch.float32)
    if binary:
        y_binary_tensor = torch.tensor(y_binary, dtype=torch.float32)
    if chrom:
        c_tensor = torch.tensor(chrom_indices, dtype=torch.long)
        if binary:
            dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor, y_binary_tensor, c_tensor)
        else:
            dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor, c_tensor)
    else:
        if binary:
            dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor, y_binary_tensor)
        else:
            dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def parser_args():
    parser = argparse.ArgumentParser(description='Train and test Autoencoder (AE) or Variational Autoencoder (VAE)')
    parser.add_argument('--model', type=str, choices=['AE', 'VAE', 'both'], default='both',
                        help='Model type to train: AE, VAE, or both (default: both)')
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
    
    # Load data
    train_input_path = input_path + filename + "train_data.npy"
    print("Loading training data from:", train_input_path)
    train_dataloader = dataloader_from_array(train_input_path, chrom=True, batch_size=64, shuffle=True)
    
    test_input_path = input_path + filename + "test_data.npy"
    print("Loading test data from:", test_input_path)
    test_dataloader = dataloader_from_array(test_input_path, chrom=True, batch_size=64, shuffle=False)
    
    
    # Create chromosome embedding once to use consistently
    chrom = True 
    if chrom:
        chrom_embedding = ChromosomeEmbedding()
    else:
        chrom_embedding = None
    
    
    # Train and test based on model choice
    if args.model in ['AE', 'both']:
        print("="*60)
        print("TRAINING AUTOENCODER (AE)")
        print("="*60)
        if args.binary:
            ae_model = AE_binary(seq_length=2000, feature_dim=8, layers=[512, 256, 128], use_conv=args.use_conv)
        else:
            ae_model = AE(seq_length=2000, feature_dim=8, layers=[512, 256, 128], use_conv=args.use_conv)
        trained_ae = train(ae_model, train_dataloader, num_epochs=10, learning_rate=1e-3, 
                          chrom=chrom, chrom_embedding=chrom_embedding, plot=True, binary=args.binary, name = filename)
        ae_reconstructions, ae_latents, ae_metrics = test(trained_ae, test_dataloader, 
                                                          chrom=True, chrom_embedding=chrom_embedding, 
                                                          plot=True, n_examples=5, binary=args.binary, name = filename)
    
    if args.model in ['VAE', 'both']:
        if args.model == 'both':
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