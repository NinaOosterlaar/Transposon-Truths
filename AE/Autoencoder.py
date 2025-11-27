import torch
import torch.nn as nn
import torch.optim as optim

class AE(nn.Module):
    def __init__(self, seq_length=2000, feature_dim=12, layers=(512, 256, 128)):
        """
        seq_length: number of positions in the window (e.g. 2000)
        feature_dim: number of features per position (e.g. 12 incl. chr embedding)
        layers: sizes of encoder layers; last one is the latent dim
        """
        super(AE, self).__init__()
        
        self.seq_length = seq_length
        self.feature_dim = feature_dim
        
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
        
       