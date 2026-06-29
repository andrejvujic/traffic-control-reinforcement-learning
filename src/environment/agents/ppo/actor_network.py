import torch.nn as nn
from torch.optim import Adam


class ActorNetwork(nn.Module):
    def __init__(self, in_features, out_features, alpha):
        super(ActorNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(
                in_features=in_features,
                out_features=256,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=256,
                out_features=256,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=256,
                out_features=out_features
            ),
        )

        self.optimizer = Adam(
            self.parameters(),
            alpha
        )

    def forward(self, x):
        return self.network(x)
