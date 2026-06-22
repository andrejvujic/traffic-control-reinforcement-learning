import torch.nn as nn


class Network(nn.Module):
    def __init__(self, in_features, out_features):
        super(Network, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(
                in_features,
                64
            ),
            nn.ReLU(),
            nn.Linear(
                64,
                128
            ),
            nn.ReLU(),
            nn.Linear(
                128,
                128
            ),
            nn.ReLU(),
            nn.Linear(
                128,
                out_features
            )
        )

    def forward(self, x):
        return self.network(x)
