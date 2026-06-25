from src.environment.agents.ppo.rollout_buffer import RolloutBuffer
from src.environment.agents.ppo.actor_network import ActorNetwork
from src.environment.agents.ppo.critic_network import CriticNetwork
from src.game.constants import TRAFFIC_LIGHT_PHASES
import torch as T
from torch.distributions import Categorical

INPUT_FEATURES = 40
OUTPUT_FEATURES = len(TRAFFIC_LIGHT_PHASES) + 1


class PPOAgent:
    def __init__(
        self,
        gamma=0.99,
        lambda_=0.95,
        epsilon=0.2,
        actor_alpha=0.0003,
        critic_alpha=0.0005
    ):
        self.memory = RolloutBuffer()

        self.actor = ActorNetwork(
            in_features=INPUT_FEATURES,
            out_features=OUTPUT_FEATURES,
            alpha=actor_alpha
        )

        self.critic = CriticNetwork(
            in_features=INPUT_FEATURES,
            alpha=critic_alpha
        )

        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon = epsilon

    def remember(self, state, action, action_log_prob, reward, value, end_flag):
        self.memory.add(
            state,
            action,
            action_log_prob,
            reward,
            value,
            end_flag
        )

    def forget(self):
        self.memory.clear()

    def calculate_advantages(self, rewards, values, end_flags, last_value=0.0):
        advantages = T.zeros(len(rewards), dtype=T.float32)
        next_advantage = 0.0

        for index in range(len(rewards) - 1, -1, -1):
            not_done = 1 - int(end_flags[index])

            if index == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[index + 1]

            delta = rewards[index] + self.gamma * next_value * not_done - values[index]
            advantages[index] = delta + self.gamma * self.lambda_ * next_advantage * not_done
            next_advantage = advantages[index]

        return advantages

    def next_action(self, state):
        with T.no_grad():
            state_tensor = T.tensor(state, dtype=T.float32)
            logits = self.actor(state_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()

        return action.item(), dist.log_prob(action).item()

    def evaluate_state(self, state):
        if not isinstance(state, T.Tensor):
            state = T.tensor(state, dtype=T.float32)
        value = self.critic(state)
        return value

    def learn(self, epochs, batch_size, last_value=0.0):
        states, actions, old_action_log_probs, rewards, values, end_flags = self.memory.get_all()

        states = T.tensor(states, dtype=T.float32)
        rewards = T.tensor(rewards, dtype=T.float32)
        values = T.tensor(values, dtype=T.float32)
        old_action_log_probs = T.tensor(old_action_log_probs, dtype=T.float32)
        actions = T.tensor(actions, dtype=T.long)

        advantages = self.calculate_advantages(rewards, values, end_flags, last_value).detach()
        returns = (advantages + values).detach()

        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        for _ in range(epochs):
            batch_indexes = self.memory.create_batches(batch_size)
            for batch in batch_indexes:
                states_tensor = states[batch]
                actions_tensor = actions[batch]

                logits = self.actor.forward(states_tensor)
                dist = Categorical(logits=logits)
                action_log_probs = dist.log_prob(actions_tensor)

                ratio = T.exp(action_log_probs - old_action_log_probs[batch])
                weighted_ratio = ratio * advantages[batch]
                weighted_ratio_clipped = T.clip(
                    ratio,
                    1.0 - self.epsilon,
                    1.0 + self.epsilon
                ) * advantages[batch]

                actor_loss = -1.0 * T.min(weighted_ratio, weighted_ratio_clipped)
                actor_loss = actor_loss.mean()

                critic_values = self.critic(states_tensor)
                critic_values = critic_values.squeeze()

                critic_loss = (returns[batch] - critic_values) ** 2
                critic_loss = critic_loss.mean()

                loss = actor_loss + critic_loss

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()

                loss.backward()

                self.actor.optimizer.step()
                self.critic.optimizer.step()

    def save(self, path):
        T.save(
            self.actor.state_dict(),
            path
        )

    def load(self, path):
        self.actor.load_state_dict(
            T.load(path)
        )
