from src.environment.agents.agent_evaluator import AgentEvaluator
from src.environment.agents.ppo.ppo_agent import PPOAgent


class PPOAgentEvaluator(AgentEvaluator):
    def __init__(self, model_path, target_games, seed):
        super().__init__(
            model_name='PPO Agent',
            target_games=target_games,
            seed=seed
        )

        self.agent = PPOAgent()
        self.agent.load(model_path)

    def select_action(self, state):
        action = self.agent.next_action(
            state,
            greedy=True
        )
        return action
