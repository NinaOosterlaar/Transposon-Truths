import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, r2_score
from results import plot_test_results, plot_training_loss
import argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

class AE(nn.Module):
    def __init__(self, seq_length=2000, feature_dim=8, layers=(512, 256, 128)):
        """
        seq_length: number of positions in the window (e.g. 2000)
        feature_dim: number of features per position (e.g. 12 incl. chr embedding)
        layers: sizes of encoder layers; last one is the latent dim
        """
        super(AE, self).__init__()
        
        self.seq_length = seq_length
        self.feature_dim = feature_dim
        self.model_type = 'AE'
        
        input_dim = seq_length * feature_dim  # e.g. 2000 * 12
        
        # ----- Encoder -----
        encoder_layers = []
        prev_dim = input_dim
        for h in layers:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.ReLU())
            prev_dim = h
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.latent_dim = layers[-1]
        
        # ----- Decoder -----
        decoder_layers = []
        prev_dim = self.latent_dim
        
        # Mirror all but the last layer in reverse (e.g. 128 -> 256 -> 512 -> input_dim)
        for h in reversed(layers[:-1]):
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(nn.ReLU())
            prev_dim = h
        
        decoder_layers.append(nn.Linear(prev_dim, seq_length))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten input
        
        # Encode
        z = self.encoder(x)
        # Decode
        x_recon = self.decoder(z)
        
        return x_recon, z


class VAE(nn.Module):
    def __init__(self, seq_length=2000, feature_dim=8, layers=(512, 256, 128)):
        """
        Variational Autoencoder
        seq_length: number of positions in the window (e.g. 2000)
        feature_dim: number of features per position (e.g. 12 incl. chr embedding)
        layers: sizes of encoder layers; last one is the latent dim
        """
        super(VAE, self).__init__()
        
        self.seq_length = seq_length
        self.feature_dim = feature_dim
        self.model_type = 'VAE'
        
        input_dim = seq_length * feature_dim
        
        # ----- Encoder -----
        encoder_layers = []
        prev_dim = input_dim
        for h in layers[:-1]:  # All but last layer
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.ReLU())
            prev_dim = h
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.latent_dim = layers[-1]
        
        # Latent space layers (mean and log variance)
        self.fc_mu = nn.Linear(prev_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, self.latent_dim)
        
        # ----- Decoder -----
        decoder_layers = []
        prev_dim = self.latent_dim
        
        # Mirror encoder in reverse
        for h in reversed(layers[:-1]):
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(nn.ReLU())
            prev_dim = h
        
        decoder_layers.append(nn.Linear(prev_dim, seq_length))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """Encode input to latent space parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + sigma * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten input
        
        # Encode
        mu, logvar = self.encode(x)
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        # Decode
        x_recon = self.decoder(z)
        
        return x_recon, z, mu, logvar
    
    
def train(model, dataloader, num_epochs=50, learning_rate=1e-3, chrom=True, plot=True, beta=1.0):
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
    plot : bool
        Whether to plot training loss
    beta : float
        Weight for KL divergence loss (only used for VAE). Default=1.0
    """
    model.to(device)
    parameters = list(model.parameters())
    if chrom:
        chrom_embedding = ChromosomeEmbedding().to(device)
        parameters += list(chrom_embedding.parameters())

    criterion = nn.MSELoss()
    optimizer = optim.Adam(parameters, lr=learning_rate)
    
    is_vae = hasattr(model, 'model_type') and model.model_type == 'VAE'
    
    epoch_losses = []
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            if chrom:
                x, y, c = batch      # dataloader has (x, y, c)
                c = c.to(device)
            else:
                x, y = batch         # dataloader has (x, y)
            
            x = x.to(device)         # (B, seq, F_other or F_other_without_chr)
            y = y.to(device)         # (B, seq)
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
                recon_loss = criterion(recon_batch, y)
                # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / y.size(0)
                loss = recon_loss + beta * kl_loss
                
                epoch_recon_loss += recon_loss.item() * y.size(0)
                epoch_kl_loss += kl_loss.item() * y.size(0)
            else:
                recon_batch, z = model(batch_input)
                loss = criterion(recon_batch, y)
            
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
        model_type_str = model.model_type if hasattr(model, 'model_type') else 'AE'
        plot_training_loss(epoch_losses, model_type=model_type_str)
    
    return model

def test(model, dataloader, chrom=True, plot=True, n_examples=5, beta=1.0):
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
    plot : bool
        Whether to create visualization plots
    n_examples : int
        Number of example reconstructions to plot
    beta : float
        Weight for KL divergence loss (only used for VAE). Default=1.0
    """
    model.to(device)
    model.eval()
    
    all_reconstructions = []
    all_latents = []
    all_originals = []
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    criterion = nn.MSELoss()
    
    is_vae = hasattr(model, 'model_type') and model.model_type == 'VAE'
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            if chrom:
                x, y, c = batch      # dataloader has (x, y, c)
                c = c.to(device)
            else:
                x, y = batch         # dataloader has (x, y)
            
            x = x.to(device)         # (B, seq, F_other or F_other_without_chr)
            y = y.to(device)         # (B, seq)
            y_in = y.unsqueeze(-1)  # Add feature dimension
            if chrom:
                chrom_embedding = ChromosomeEmbedding().to(device)
                c_emb = chrom_embedding(c)
                batch_input = torch.cat((y_in, x, c_emb), dim=2)
            else:
                batch_input = torch.cat((y_in, x), dim=2)
            
            # Forward pass
            if is_vae:
                recon_batch, z, mu, logvar = model(batch_input)
                # Calculate losses
                recon_loss = criterion(recon_batch, y)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / y.size(0)
                loss = recon_loss + beta * kl_loss
                
                total_recon_loss += recon_loss.item() * y.size(0)
                total_kl_loss += kl_loss.item() * y.size(0)
            else:
                recon_batch, z = model(batch_input)
                loss = criterion(recon_batch, y)
            
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
        model_type_str = model.model_type if hasattr(model, 'model_type') else 'AE'
        plot_test_results(all_originals, all_reconstructions, model_type=model_type_str, 
                         n_examples=n_examples, metrics=metrics)
    
    return all_reconstructions, all_latents, metrics


# Embed the chromosome feature if needed
class ChromosomeEmbedding(nn.Module):
    def __init__(self, num_chromosomes=17, embedding_dim=4):
        super(ChromosomeEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_chromosomes, embedding_dim)
        
    def forward(self, x):
        return self.embedding(x)

def dataloader_from_array(input, chrom=True, batch_size=64, shuffle=True):
    data_array = np.load(input)
    counts = data_array[:, :, 0]
    if chrom:
        features = data_array[:, :, 1:-1]
        chrom_indices = data_array[:, :, -1].astype(np.int64)
    else:
        features = data_array[:, :, 1:]
    
    x_tensor = torch.tensor(features, dtype=torch.float32)
    y_tensor = torch.tensor(counts, dtype=torch.float32)
    if chrom:
        c_tensor = torch.tensor(chrom_indices, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor, c_tensor)
    else:
        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

    
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and test Autoencoder (AE) or Variational Autoencoder (VAE)')
    parser.add_argument('--model', type=str, choices=['AE', 'VAE', 'both'], default='both',
                        help='Model type to train: AE, VAE, or both (default: both)')
    
    args = parser.parse_args()
    
    # Load data
    input_path = "Data/processed_data/train_data.npy"
    train_dataloader = dataloader_from_array(input_path, chrom=True, batch_size=64, shuffle=True)
    
    test_input_path = "Data/processed_data/test_data.npy"
    test_dataloader = dataloader_from_array(test_input_path, chrom=True, batch_size=64, shuffle=False)
    
    # Train and test based on model choice
    if args.model in ['AE', 'both']:
        print("="*60)
        print("TRAINING AUTOENCODER (AE)")
        print("="*60)
        ae_model = AE(seq_length=2000, feature_dim=8, layers=[512, 256, 128])
        trained_ae = train(ae_model, train_dataloader, num_epochs=10, learning_rate=1e-3, chrom=True, plot=True)
        ae_reconstructions, ae_latents, ae_metrics = test(trained_ae, test_dataloader, chrom=True, plot=True, n_examples=5)
    
    if args.model in ['VAE', 'both']:
        if args.model == 'both':
            print("\n" + "="*60)
        else:
            print("="*60)
        print("TRAINING VARIATIONAL AUTOENCODER (VAE)")
        print("="*60)
        vae_model = VAE(seq_length=2000, feature_dim=8, layers=[512, 256, 128])
        trained_vae = train(vae_model, train_dataloader, num_epochs=10, learning_rate=1e-3, chrom=True, plot=True, beta=1.0)
        vae_reconstructions, vae_latents, vae_metrics = test(trained_vae, test_dataloader, chrom=True, plot=True, n_examples=5, beta=1.0)
    
    