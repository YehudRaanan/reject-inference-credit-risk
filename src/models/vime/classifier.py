"""
Classification heads for VIME models.

Attaches to the pre-trained VIME encoder's latent vector
and outputs P(default).
"""
import torch
import torch.nn as nn


class ClassificationHead128(nn.Module):
    """
    Classification head for VIME-128.

    Architecture:
        [z] (128) -> Linear(128->256) -> BN -> ReLU -> Dropout(0.3)
                  -> Linear(256->64)  -> BN -> ReLU -> Dropout(0.3)
                  -> Linear(64->1)    -> Sigmoid -> P(default)
    """

    def __init__(self, embedding_dim=128, dropout=0.3):
        super().__init__()

        self.fc1 = nn.Linear(embedding_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(256, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(64, 1)

    def forward(self, z):
        """
        Args:
            z: VIME encoder output (batch_size, 128)
        Returns:
            P(default) as tensor (batch_size, 1)
        """
        z = self.drop1(self.relu1(self.bn1(self.fc1(z))))
        z = self.drop2(self.relu2(self.bn2(self.fc2(z))))
        return torch.sigmoid(self.fc3(z))


class ClassificationHead512(nn.Module):
    """
    Classification head for VIME-512.

    Architecture:
        [z] (512) -> Linear(512->256) -> BN -> ReLU -> Dropout(0.3)
                  -> Linear(256->64)  -> BN -> ReLU -> Dropout(0.3)
                  -> Linear(64->1)    -> Sigmoid -> P(default)
    """

    def __init__(self, embedding_dim=512, dropout=0.3):
        super().__init__()

        self.fc1 = nn.Linear(embedding_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(256, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(64, 1)

    def forward(self, z):
        """
        Args:
            z: VIME encoder output (batch_size, 512)
        Returns:
            P(default) as tensor (batch_size, 1)
        """
        z = self.drop1(self.relu1(self.bn1(self.fc1(z))))
        z = self.drop2(self.relu2(self.bn2(self.fc2(z))))
        return torch.sigmoid(self.fc3(z))
