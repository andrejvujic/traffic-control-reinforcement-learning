from model_evaluation import ModelEvaluation
from src.environment.agents.dqn.dqn_agent import DQNAgent


class DQNAgentEvaluation(ModelEvaluation):
    def __init__(self, model_path, target_games, seed):
        super().__init__(
            model_name='DQN Agent',
            target_games=target_games,
            seed=seed
        )

        self.agent = DQNAgent()
        self.agent.load(model_path)

    def select_action(self, state):
        action = self.agent.next_action(
            state,
            epsilon=0.0
        )
        return action
