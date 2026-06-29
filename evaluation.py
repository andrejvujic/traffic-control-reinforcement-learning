from src.environment.agents.ppo.ppo_evaluation import PPOAgentEvaluation
from src.environment.agents.dqn.dqn_evaluation import DQNAgentEvaluation
from src.environment.agents.basic.basic_evaluation import BasicAgentEvaluation
from src.environment.agents.random.random_evaluation import RandomAgentEvaluation

EVALUATION_GAMES = 1000
EVALUATION_SEED = 42

random_evaluation = RandomAgentEvaluation(
    target_games=EVALUATION_GAMES,
    seed=EVALUATION_SEED
)

# random_evaluation.run()
random_evaluation.end()

print('\n\n\n')

basic_evaluation = BasicAgentEvaluation(
    target_games=EVALUATION_GAMES,
    seed=EVALUATION_SEED
)

# basic_evaluation.run()
basic_evaluation.end()

print('\n\n\n')

dqn_evaluation = DQNAgentEvaluation(
    model_path='training_output/dqn/1782752102/model.pt',
    target_games=EVALUATION_GAMES,
    seed=EVALUATION_SEED
)

dqn_evaluation.run()
dqn_evaluation.end()

print('\n\n\n')

ppo_evaluation = PPOAgentEvaluation(
    model_path='training_output/ppo/1782736882/model.pt',
    target_games=EVALUATION_GAMES,
    seed=EVALUATION_SEED
)

ppo_evaluation.run()
ppo_evaluation.end()

print('\n\n\n')
