from ppo_evaluation import PPOAgentEvaluation
from basic_evaluation import BasicAgentEvaluation
from random_evaluation import RandomAgentEvaluation

EVALUATION_GAMES = 1000
EVALUATION_SEED = 42

random_evaluation = RandomAgentEvaluation(
    target_games=EVALUATION_GAMES,
    seed=EVALUATION_SEED
)

random_evaluation.run()
random_evaluation.end()

print('\n\n\n')


basic_evaluation = BasicAgentEvaluation(
    target_games=EVALUATION_GAMES,
    seed=EVALUATION_SEED
)

basic_evaluation.run()
basic_evaluation.end()

print('\n\n\n')


ppo_evaluation = PPOAgentEvaluation(
    model_path='training_output/ppo/1782405690/model.pt',
    target_games=EVALUATION_GAMES,
    seed=EVALUATION_SEED
)

ppo_evaluation.run()
ppo_evaluation.end()

print('\n\n\n')
