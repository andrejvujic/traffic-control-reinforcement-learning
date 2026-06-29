from src.environment.agents.agent_evaluation import AgentEvaluation
from src.environment.agents.dqn.dqn_agent import DQNAgent


class DQNAgentEvaluation(AgentEvaluation):
    def __init__(self, model_path, target_games, seed):
        super().__init__(
            model_name='DQN Agent',
            target_games=target_games,
            seed=seed
        )

        self.agent = DQNAgent()
        self.agent.load(model_path)

    def select_action(self, state):
        return self.agent.next_action(
            state,
            epsilon=0.0
        )
