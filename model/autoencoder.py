"""Symmetric feed-forward autoencoder for unsupervised anomaly detection."""
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """Encoder 57->64->32->16 (bottleneck), decoder mirrors back to input_dim.

    BatchNorm + ReLU + Dropout in the hidden layers; Sigmoid output to match the
    [0, 1] range of Min-Max scaled inputs. Xavier-uniform initialisation.
    """

    def __init__(self, input_dim: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 16),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, input_dim), nn.Sigmoid(),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.decoder(self.encoder(x))
