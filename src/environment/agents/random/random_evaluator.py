from src.environment.agents.agent_evaluator import AgentEvaluator
from src.environment.agents.random.random_agent import RandomAgent


class RandomAgentEvaluator(AgentEvaluator):
    def __init__(self, target_games, seed):
        super().__init__(
            model_name='Random Agent',
            target_games=target_games,
            seed=seed
        )

        self.agent = RandomAgent()

    def select_action(self, _):
        return self.agent.next_action(_)
