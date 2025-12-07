import torch
import torch.nn as nn

class ZINBAE(nn.Module):
    def __init__(
        self,
        seq_length=2000,
        feature_dim=8,
        layers=(512, 256, 128),
        use_conv=False,
        conv_channels=64,
        pool_size=2,
        kernel_size=3,
        padding=1,
        stride=1,
    ):
        super().__init__()
        
        self.seq_length = seq_length
        self.feature_dim = feature_dim
        self.model_type = 'ZINBAE'
        self.use_conv = use_conv
        
        # ----- Optional Conv1D Layer -----
        if use_conv:
            self.conv1d = nn.Conv1d(
                in_channels=feature_dim,
                out_channels=conv_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            self.conv_relu = nn.ReLU()
            self.pool = nn.MaxPool1d(kernel_size=pool_size)
            pooled_seq_length = seq_length // pool_size
            input_dim = pooled_seq_length * conv_channels
        else:
            input_dim = seq_length * feature_dim
        
        # ----- Encoder -----
        encoder_layers = []
        prev_dim = input_dim
        for h in layers:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.ReLU())
            prev_dim = h
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.latent_dim = layers[-1]
        
        # ----- Decoder "body" (shared) -----
        decoder_layers = []
        prev_dim = self.latent_dim
        
        # mirror all but last
        for h in reversed(layers[:-1]):
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(nn.ReLU())
            prev_dim = h
        
        # this shared decoder output D will feed μ, θ, π heads
        self.decoder_shared = nn.Sequential(*decoder_layers)
        decoder_out_dim = prev_dim  # last h in loop above
        
        # ----- ZINB heads -----
        # Each outputs seq_length parameters (one per position)
        self.mu_layer    = nn.Linear(decoder_out_dim, seq_length)
        self.theta_layer = nn.Linear(decoder_out_dim, seq_length)
        self.pi_layer    = nn.Linear(decoder_out_dim, seq_length)
    
    def forward(self, x_in, size_factors):
        """
        x_in: preprocessed input (e.g. log1p(CPM), maybe scaled)
              shape (batch, seq_length, feature_dim)
        size_factors: library-size factors for each sample
              shape (batch,) or (batch, 1)
        """
        batch_size = x_in.size(0)
        
        if self.use_conv:
            # (batch, seq_length, feature_dim) -> (batch, feature_dim, seq_length)
            x = x_in.permute(0, 2, 1)
            x = self.conv1d(x)
            x = self.conv_relu(x)
            x = self.pool(x)  # (batch, conv_channels, pooled_seq_length)
            x = x.permute(0, 2, 1).contiguous()
        else:
            x = x_in
        
        x = x.view(batch_size, -1)  # flatten
        
        # Encode
        z = self.encoder(x)
        
        # Decode shared representation
        D = self.decoder_shared(z)  # shape (batch, decoder_out_dim)
        
        # ZINB parameters
        mu_hat = torch.exp(self.mu_layer(D))       # (batch, seq_length), positive
        theta  = torch.exp(self.theta_layer(D))    # (batch, seq_length), positive
        pi     = torch.sigmoid(self.pi_layer(D))   # (batch, seq_length), in (0,1)
        
        # apply size factors to μ
        if size_factors.dim() == 1:
            size_factors = size_factors.unsqueeze(1)  # (batch, 1)
        mu = mu_hat * size_factors                   # broadcast over seq_length
        
        return mu, theta, pi, z
    
class ZINBVAE(nn.Module):
    def __init__(
        self,
        seq_length=2000,
        feature_dim=8,
        layers=(512, 256, 128),
        use_conv=False,
        conv_channels=64,
        pool_size=2,
        kernel_size=3,
        padding=1,
        stride=1,
    ):
        super().__init__()
        
        self.seq_length = seq_length
        self.feature_dim = feature_dim
        self.model_type = 'ZINBVAE'
        self.use_conv = use_conv
        
        # ----- Optional Conv1D Layer -----
        if self.use_conv:
            self.conv1d = nn.Conv1d(
                in_channels=feature_dim,
                out_channels=conv_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            self.conv_relu = nn.ReLU()
            self.pool = nn.MaxPool1d(kernel_size=pool_size)
            pooled_seq_length = seq_length // pool_size
            input_dim = pooled_seq_length * conv_channels
        else:
            input_dim = seq_length * feature_dim
            
        # ----- Encoder -----
        encoder_layers = []
        prev_dim = input_dim
        for h in layers:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.ReLU())
            prev_dim = h
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.latent_dim = layers[-1]
        
        # Latent space layers
        self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.latent_dim, self.latent_dim)
        
        # ----- Decoder -----
        decoder_layers = []
        prev_dim = self.latent_dim
        
        # mirror all but last
        for h in reversed(layers[:-1]):
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(nn.ReLU())
            prev_dim = h
            
        # this shared decoder output D will feed μ, θ, π heads
        self.decoder_shared = nn.Sequential(*decoder_layers)
        decoder_out_dim = prev_dim  # last h in loop above
        
        # ----- ZINB heads -----
        # Each outputs seq_length parameters (one per position)
        self.mu_layer    = nn.Linear(decoder_out_dim, seq_length)
        self.theta_layer = nn.Linear(decoder_out_dim, seq_length)
        self.pi_layer    = nn.Linear(decoder_out_dim, seq_length)
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x_in, size_factors):
        batch_size = x_in.size(0)
        
        if self.use_conv:
            # (batch, seq_length, feature_dim) -> (batch, feature_dim, seq_length)
            x = x_in.permute(0, 2, 1)
            x = self.conv1d(x)
            x = self.conv_relu(x)
            x = self.pool(x)  # (batch, conv_channels, pooled_seq_length)
            x = x.permute(0, 2, 1).contiguous()
        else:
            x = x_in
            
        x = x.view(batch_size, -1)  # flatten
            
        # Encode
        mu_z, logvar_z = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        
        D = self.decoder_shared(z)  # shape (batch, decoder_out_dim)
        
        # ZINB parameters
        mu_hat = torch.exp(self.mu_layer(D))       # (batch, seq_length), positive
        theta  = torch.exp(self.theta_layer(D))    # (batch, seq_length), positive
        pi     = torch.sigmoid(self.pi_layer(D))   # (batch, seq_length), in (0,1)
        
        # apply size factors to μ
        if size_factors.dim() == 1:
            size_factors = size_factors.unsqueeze(1)  # (batch, 1)
        mu = mu_hat * size_factors                   # broadcast over seq_length
        
        return mu, theta, pi, z, mu_z, logvar_z
        