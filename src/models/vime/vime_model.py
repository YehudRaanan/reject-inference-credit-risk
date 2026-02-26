import torch
import torch.nn as nn
import config

class VIME(nn.Module):
    """
    VIME (Value Imputation and Mask Estimation) Autoencoder.

    Architecture:
    - Encoder: Input (d) -> Hidden (d) -> Embedding (z)
    - Mask Estimator: Embedding (z) -> Mask Prediction (d) [Sigmoid]
    - Feature Reconstructor: Embedding (z) -> Feature Reconstruction (d)

    Reference: Jinsung Yoon et al., "VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain", NeurIPS 2020.
    """
    def __init__(self, input_dim=328, embedding_dim=config.VIME_EMBEDDING_DIM, hidden_dim=config.VIME_HIDDEN_DIM, dropout=0.1):
        super(VIME, self).__init__()

        # --- Encoder Layers [cite: 35] ---
        # Layer 1: 328 -> 256
        self.enc_fc1 = nn.Linear(input_dim, hidden_dim)
        self.enc_relu1 = nn.ReLU()

        # Layer 2: 256 -> 256
        self.enc_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.enc_relu2 = nn.ReLU()

        # Dropout for regularization
        self.enc_dropout = nn.Dropout(dropout)

        # Layer 3: Latent/Embedding layer (256 -> 128)
        self.enc_fc3 = nn.Linear(hidden_dim, embedding_dim)
        
        # --- Mask Estimator Head  ---
        # Predicts which features were corrupted (128 -> 328)
        self.mask_fc = nn.Linear(embedding_dim, input_dim)
        
        # --- Feature Reconstructor Head  ---
        # Reconstructs the original raw values (128 -> 328)
        self.recon_fc = nn.Linear(embedding_dim, input_dim)

    def encode(self, x):
        """Returns the latent vector (z)."""
        x = self.enc_relu1(self.enc_fc1(x))
        x = self.enc_relu2(self.enc_fc2(x))
        x = self.enc_dropout(x)
        latent_vector = self.enc_fc3(x)
        return latent_vector

    def forward(self, x):
        # Encoding phase [cite: 35]
        latent_vector = self.encode(x)
        
        # Mask Estimation 
        # Predict the binary mask vector of corrupted/hidden inputs 
        m_hat = torch.sigmoid(self.mask_fc(latent_vector))
        
        # Feature Reconstruction 
        # Reconstruct the original raw feature values 
        x_hat = self.recon_fc(latent_vector)
        
        return m_hat, x_hat
