from model_evaluation import ModelEvaluation
from src.environment.agents.basic.basic_agent import BasicAgent
import torch as T


class BasicAgentEvaluation(ModelEvaluation):
    def __init__(self, target_games, seed):
        super().__init__(
            model_name='Basic Agent',
            target_games=target_games,
            seed=seed
        )

        self.agent = BasicAgent()

    def select_action(self, state):
        action = self.agent.next_action(
            state,
        )
        return action
