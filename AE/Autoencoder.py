import torch
import torch.nn as nn

class AE(nn.Module):
    def __init__(self, seq_length=2000, feature_dim=8, layers=(512, 256, 128), use_conv=False, conv_channels=64, pool_size=2, kernel_size=3, padding=1, stride=1):
        """
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
        self.model_type = 'AE'
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
        
        decoder_layers.append(nn.Linear(prev_dim, seq_length))
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
        Variational Autoencoder
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
        self.model_type = 'VAE'
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
    
    

    
    