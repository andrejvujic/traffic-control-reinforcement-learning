import numpy as np
from src.game.constants import TRAFFIC_LIGHT_PHASES
from src.environment.agents.agent import Agent

ACTIONS = range(
    len(TRAFFIC_LIGHT_PHASES) + 1
)


class RandomAgent(Agent):
    def __init__(self, seed=None):
        super().__init__(name='Random')
        np.random.seed(seed)

    def next_action(self, _):
        return np.random.choice(ACTIONS)
