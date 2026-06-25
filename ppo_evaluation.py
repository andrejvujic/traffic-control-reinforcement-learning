from model_evaluation import ModelEvaluation
from src.environment.agents.ppo.ppo_agent import PPOAgent
import torch as T


class PPOAgentEvaluation(ModelEvaluation):
    def __init__(self, model_path, target_games, seed):
        super().__init__(
            model_name='PPO Agent',
            target_games=target_games,
            seed=seed
        )

        self.agent = PPOAgent()
        self.agent.load(model_path)

    def select_action(self, state):
        action, _ = self.agent.next_action(
            state,
            deterministc=True
        )
        return action
