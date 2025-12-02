import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from results import plot_binary_test_results, plot_binary_training_loss
import argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

class AE(nn.Module):
    def __init__(self, seq_length=2000, feature_dim=8, layers=(512, 256, 128), use_conv=False, conv_channels=64, pool_size=2, kernel_size=3, padding=1, stride=1):
        """
        Binary Autoencoder for presence/absence prediction
        seq_length: number of positions in the window (e.g. 2000)
        feature_dim: number of features per position (e.g. 12 incl. chr embedding)
        layers: sizes of encoder layers; last one is the latent dim
        use_conv: whether to use Conv1D as the first layer
        conv_channels: number of output channels for the Conv1D layer (default: 64)
        pool_size: pooling kernel size to reduce sequence length (default: 2)
        """
        super(AE, self).__init__()
        
        self.seq_length = seq_length
        self.feature_dim = feature_dim
        self.model_type = 'AE_binary'
        self.use_conv = use_conv
        
        # ----- Optional Conv1D Layer -----
        if use_conv:
            self.conv1d = nn.Conv1d(in_channels=feature_dim, out_channels=conv_channels, 
                                   kernel_size=kernel_size, stride=stride, padding=padding)
            self.conv_relu = nn.ReLU()
            self.pool = nn.MaxPool1d(kernel_size=pool_size)
            pooled_seq_length = seq_length // pool_size
            input_dim = pooled_seq_length * conv_channels  # Flattened after conv + pool
        else:
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
        
        # Final layer with sigmoid for binary prediction
        decoder_layers.append(nn.Linear(prev_dim, seq_length))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        if self.use_conv:
            # x shape: (batch, seq_length, feature_dim)
            # Conv1D expects: (batch, feature_dim, seq_length)
            x = x.permute(0, 2, 1)  # Rearrange to (batch, feature_dim, seq_length)
            x = self.conv1d(x)       # Apply Conv1D -> (batch, conv_channels, seq_length)
            x = self.conv_relu(x)    # Apply ReLU
            x = self.pool(x)         # Apply MaxPool -> (batch, conv_channels, pooled_seq_length)
            x = x.permute(0, 2, 1).contiguous()   # Back to (batch, pooled_seq_length, conv_channels)
        
        x = x.view(batch_size, -1)  # Flatten input
        
        # Encode
        z = self.encoder(x)
        # Decode
        x_recon = self.decoder(z)
        
        return x_recon, z


class VAE(nn.Module):
    def __init__(self, seq_length=2000, feature_dim=8, layers=(512, 256, 128), use_conv=False, conv_channels=64, pool_size=2, kernel_size=3, padding=1, stride=1):
        """
        Binary Variational Autoencoder for presence/absence prediction
        seq_length: number of positions in the window (e.g. 2000)
        feature_dim: number of features per position (e.g. 12 incl. chr embedding)
        layers: sizes of encoder layers; last one is the latent dim
        use_conv: whether to use Conv1D as the first layer
        conv_channels: number of output channels for the Conv1D layer (default: 64)
        pool_size: pooling kernel size to reduce sequence length (default: 2)
        """
        super(VAE, self).__init__()
        
        self.seq_length = seq_length
        self.feature_dim = feature_dim
        self.model_type = 'VAE_binary'
        self.use_conv = use_conv
        
        # ----- Optional Conv1D Layer -----
        if use_conv:
            self.conv1d = nn.Conv1d(in_channels=feature_dim, out_channels=conv_channels, 
                                   kernel_size=kernel_size, stride=stride, padding=padding)
            self.conv_relu = nn.ReLU()
            self.pool = nn.MaxPool1d(kernel_size=pool_size)
            pooled_seq_length = seq_length // pool_size
            input_dim = pooled_seq_length * conv_channels  # Flattened after conv + pool
        else:
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
        
        # Final layer with sigmoid for binary prediction
        decoder_layers.append(nn.Linear(prev_dim, seq_length))
        decoder_layers.append(nn.Sigmoid())
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
        
        if self.use_conv:
            # x shape: (batch, seq_length, feature_dim)
            # Conv1D expects: (batch, feature_dim, seq_length)
            x = x.permute(0, 2, 1)  # Rearrange to (batch, feature_dim, seq_length)
            x = self.conv1d(x)       # Apply Conv1D -> (batch, conv_channels, seq_length)
            x = self.conv_relu(x)    # Apply ReLU
            x = self.pool(x)         # Apply MaxPool -> (batch, conv_channels, pooled_seq_length)
            x = x.permute(0, 2, 1).contiguous()   # Back to (batch, pooled_seq_length, conv_channels)
        
        x = x.view(batch_size, -1)  # Flatten input
        
        # Encode
        mu, logvar = self.encode(x)
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        # Decode
        x_recon = self.decoder(z)
        
        return x_recon, z, mu, logvar
    
    
def train(model, dataloader, num_epochs=50, learning_rate=1e-3, chrom=True, chrom_embedding=None, plot=True, beta=1.0):
    """
    Train binary AE or VAE model
    
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
    """
    model.to(device)
    parameters = list(model.parameters())
    if chrom:
        if chrom_embedding is None:
            raise ValueError("chrom_embedding must be provided when chrom=True")
        chrom_embedding.to(device)
        parameters += list(chrom_embedding.parameters())

    criterion = nn.BCELoss(reduction='sum')  # Use sum reduction for proper scaling with KL loss
    optimizer = optim.Adam(parameters, lr=learning_rate)
    
    is_vae = hasattr(model, 'model_type') and model.model_type == 'VAE_binary'
    
    epoch_losses = []
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            if chrom:
                x, y, y_binary, c = batch      # dataloader has (x, y, y_binary, c)
                c = c.to(device)
            else:
                x, y, y_binary = batch         # dataloader has (x, y, y_binary)
            
            x = x.to(device)         # (B, seq, F_other or F_other_without_chr)
            y = y.to(device)         # (B, seq) - original counts for input
            y_binary = y_binary.to(device)  # (B, seq) - binary targets
            
            optimizer.zero_grad()
            y_in = y.unsqueeze(-1)  # Add feature dimension - use original counts
            if chrom:
                c_emb = chrom_embedding(c)
                batch_input = torch.cat((y_in, x, c_emb), dim=2)
            else:
                batch_input = torch.cat((y_in, x), dim=2)
            
            # Forward pass
            if is_vae:
                recon_batch, z, mu, logvar = model(batch_input)
                # VAE loss = BCE loss + KL divergence
                recon_loss = criterion(recon_batch, y_binary) / y_binary.size(0)  # Divide by batch size
                # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / y_binary.size(0)
                loss = recon_loss + beta * kl_loss
                
                epoch_recon_loss += recon_loss.item() * y_binary.size(0)
                epoch_kl_loss += kl_loss.item() * y_binary.size(0)
            else:
                recon_batch, z = model(batch_input)
                loss = criterion(recon_batch, y_binary) / y_binary.size(0)  # Divide by batch size for consistency
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * y_binary.size(0)
            
            # Update progress bar with current loss
            if is_vae:
                pbar.set_postfix({'total_loss': f'{loss.item():.4f}', 
                                 'bce': f'{recon_loss.item():.4f}',
                                 'kl': f'{kl_loss.item():.4f}'})
            else:
                pbar.set_postfix({'bce_loss': f'{loss.item():.4f}'})
        
        epoch_loss /= len(dataloader.dataset)
        epoch_losses.append(epoch_loss)
        
        if is_vae:
            epoch_recon_loss /= len(dataloader.dataset)
            epoch_kl_loss /= len(dataloader.dataset)
            print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {epoch_loss:.4f}, "
                  f"BCE: {epoch_recon_loss:.4f}, KL: {epoch_kl_loss:.4f}")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}], BCE Loss: {epoch_loss:.4f}")
    
    if plot:
        model_type_str = model.model_type if hasattr(model, 'model_type') else 'AE_binary'
        use_conv = model.use_conv if hasattr(model, 'use_conv') else False
        plot_binary_training_loss(epoch_losses, model_type=model_type_str, use_conv=use_conv)
    
    return model

def test(model, dataloader, chrom=True, chrom_embedding=None, plot=True, n_examples=5, beta=1.0, threshold=0.5):
    """
    Test binary AE or VAE model
    
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
    threshold : float
        Threshold for converting probabilities to binary predictions (default: 0.5)
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
    all_probabilities = []
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    criterion = nn.BCELoss(reduction='sum')  # Use sum reduction for proper scaling
    
    is_vae = hasattr(model, 'model_type') and model.model_type == 'VAE_binary'
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            if chrom:
                x, y, y_binary, c = batch      # dataloader has (x, y, y_binary, c)
                c = c.to(device)
            else:
                x, y, y_binary = batch         # dataloader has (x, y, y_binary)
            
            x = x.to(device)         # (B, seq, F_other or F_other_without_chr)
            y = y.to(device)         # (B, seq) - original counts for input
            y_binary = y_binary.to(device)  # (B, seq) - binary targets
            
            y_in = y.unsqueeze(-1)  # Add feature dimension - use original counts
            if chrom:
                c_emb = chrom_embedding(c)
                batch_input = torch.cat((y_in, x, c_emb), dim=2)
            else:
                batch_input = torch.cat((y_in, x), dim=2)
            
            # Forward pass
            if is_vae:
                recon_batch, z, mu, logvar = model(batch_input)
                # Calculate losses
                recon_loss = criterion(recon_batch, y_binary) / y_binary.size(0)  # Divide by batch size
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / y_binary.size(0)
                loss = recon_loss + beta * kl_loss
                
                total_recon_loss += recon_loss.item() * y_binary.size(0)
                total_kl_loss += kl_loss.item() * y_binary.size(0)
            else:
                recon_batch, z = model(batch_input)
                loss = criterion(recon_batch, y_binary) / y_binary.size(0)  # Divide by batch size
            
            total_loss += loss.item() * y_binary.size(0)
            
            all_probabilities.append(recon_batch.cpu().numpy())
            all_latents.append(z.cpu().numpy())
            all_originals.append(y_binary.cpu().numpy())
    
    all_probabilities = np.concatenate(all_probabilities, axis=0)
    all_latents = np.concatenate(all_latents, axis=0)
    all_originals = np.concatenate(all_originals, axis=0)
    
    # Convert probabilities to binary predictions
    all_reconstructions = (all_probabilities >= threshold).astype(float)
    
    # Calculate metrics
    test_loss = total_loss / len(all_originals)
    
    # Binary classification metrics
    accuracy = accuracy_score(all_originals.flatten(), all_reconstructions.flatten())
    precision = precision_score(all_originals.flatten(), all_reconstructions.flatten(), zero_division=0)
    recall = recall_score(all_originals.flatten(), all_reconstructions.flatten(), zero_division=0)
    f1 = f1_score(all_originals.flatten(), all_reconstructions.flatten(), zero_division=0)
    
    # AUC-ROC (using probabilities)
    try:
        auc_roc = roc_auc_score(all_originals.flatten(), all_probabilities.flatten())
    except ValueError:
        auc_roc = 0.0  # In case only one class is present
    
    # Build metrics dictionary
    metrics = {
        'bce_loss': test_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc
    }
    
    if is_vae:
        test_recon_loss = total_recon_loss / len(all_originals)
        test_kl_loss = total_kl_loss / len(all_originals)
        metrics['recon_loss'] = test_recon_loss
        metrics['kl_loss'] = test_kl_loss
    
    # Print metrics
    print("\n" + "="*50)
    print("TEST RESULTS (Binary Classification)")
    print("="*50)
    print(f"Test Loss (BCE): {test_loss:.6f}")
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1 Score: {f1:.6f}")
    print(f"AUC-ROC: {auc_roc:.6f}")
    
    if is_vae:
        print(f"Reconstruction Loss (BCE): {test_recon_loss:.6f}")
        print(f"KL Divergence: {test_kl_loss:.6f}")
    
    print("="*50 + "\n")
    
    if plot:
        model_type_str = model.model_type if hasattr(model, 'model_type') else 'AE_binary'
        use_conv = model.use_conv if hasattr(model, 'use_conv') else False
        plot_binary_test_results(all_originals, all_reconstructions, all_probabilities,
                                model_type=model_type_str, n_examples=n_examples, metrics=metrics, use_conv=use_conv)
    
    return all_reconstructions, all_latents, metrics


# Embed the chromosome feature if needed
class ChromosomeEmbedding(nn.Module):
    def __init__(self, num_chromosomes=17, embedding_dim=4):
        super(ChromosomeEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_chromosomes, embedding_dim)
        
    def forward(self, x):
        return self.embedding(x)

def dataloader_from_array(input, chrom=True, batch_size=64, shuffle=True):
    """
    Create dataloader with binary targets
    
    Parameters:
    -----------
    input : str
        Path to .npy file containing data
    chrom : bool
        Whether to include chromosome information
    batch_size : int
        Batch size for dataloader
    shuffle : bool
        Whether to shuffle data
    
    Returns:
    --------
    DataLoader with (features, original_counts, binary_counts, [chromosome_indices])
    """
    data_array = np.load(input)
    counts = data_array[:, :, 0]
    
    # Create binary targets (0 if count=0, 1 if count>0)
    binary_counts = (counts > 0).astype(np.float32)
    
    if chrom:
        features = data_array[:, :, 1:-1]
        chrom_indices = data_array[:, :, -1].astype(np.int64)
    else:
        features = data_array[:, :, 1:]
    
    x_tensor = torch.tensor(features, dtype=torch.float32)
    y_tensor = torch.tensor(counts, dtype=torch.float32)  # Original counts for input
    y_binary_tensor = torch.tensor(binary_counts, dtype=torch.float32)  # Binary targets
    
    if chrom:
        c_tensor = torch.tensor(chrom_indices, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor, y_binary_tensor, c_tensor)
    else:
        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor, y_binary_tensor)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

    
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and test Binary Autoencoder (AE) or Binary Variational Autoencoder (VAE)')
    parser.add_argument('--model', type=str, choices=['AE', 'VAE', 'both'], default='both',
                        help='Model type to train: AE, VAE, or both (default: both)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary classification (default: 0.5)')
    parser.add_argument('--use_conv', action='store_true',
                        help='Whether to use Conv1D layer in the model')
    
    args = parser.parse_args()
    
    # Load data
    input_path = "Data/processed_data/train_data.npy"
    train_dataloader = dataloader_from_array(input_path, chrom=True, batch_size=64, shuffle=True)
    
    test_input_path = "Data/processed_data/test_data.npy"
    test_dataloader = dataloader_from_array(test_input_path, chrom=True, batch_size=64, shuffle=False)
    
    # Create chromosome embedding once to use consistently
    chrom_embedding = ChromosomeEmbedding()
    
    # Train and test based on model choice
    if args.model in ['AE', 'both']:
        print("="*60)
        print("TRAINING BINARY AUTOENCODER (AE)")
        print("="*60)
        ae_model = AE(seq_length=2000, feature_dim=8, layers=[512, 256, 128], use_conv=args.use_conv)
        trained_ae = train(ae_model, train_dataloader, num_epochs=10, learning_rate=1e-3, 
                          chrom=True, chrom_embedding=chrom_embedding, plot=True)
        ae_reconstructions, ae_latents, ae_metrics = test(trained_ae, test_dataloader, 
                                                          chrom=True, chrom_embedding=chrom_embedding,
                                                          plot=True, n_examples=5, threshold=args.threshold)
    
    if args.model in ['VAE', 'both']:
        if args.model == 'both':
            print("\n" + "="*60)
        else:
            print("="*60)
        print("TRAINING BINARY VARIATIONAL AUTOENCODER (VAE)")
        print("="*60)
        vae_model = VAE(seq_length=2000, feature_dim=8, layers=[512, 256, 128])
        trained_vae = train(vae_model, train_dataloader, num_epochs=10, learning_rate=1e-3, 
                           chrom=True, chrom_embedding=chrom_embedding, plot=True, beta=1.0)
        vae_reconstructions, vae_latents, vae_metrics = test(trained_vae, test_dataloader, 
                                                             chrom=True, chrom_embedding=chrom_embedding,
                                                             plot=True, n_examples=5, beta=1.0, threshold=args.threshold)
