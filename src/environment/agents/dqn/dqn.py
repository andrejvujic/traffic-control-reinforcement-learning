import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, in_features, out_features):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(
                in_features,
                256
            ),
            nn.ReLU(),
            nn.Linear(
                256,
                256
            ),
            nn.ReLU(),
            nn.Linear(
                256,
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
