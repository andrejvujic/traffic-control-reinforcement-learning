import torch.nn as nn
from torch.optim import Adam
import torch as T


class CriticNetwork(nn.Module):
    def __init__(self, in_features, alpha):
        super(CriticNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(
                in_features,
                256,
            ),
            nn.ReLU(),
            nn.Linear(
                256,
                256,
            ),
            nn.ReLU(),
            nn.Linear(
                256,
                1
            )
        )

        self.optimizer = Adam(self.parameters(), alpha)

    def forward(self, x):
        return self.network(x)
