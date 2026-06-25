import numpy as np
from src.game.constants import TRAFFIC_LIGHT_PHASES


class RandomAgent:
    def __init__(self, seed=None):
        np.random.seed(seed)

    def next_action(self):
        return np.random.choice(self.ACTIONS)

    ACTIONS = range(
        len(TRAFFIC_LIGHT_PHASES) + 1
    )
