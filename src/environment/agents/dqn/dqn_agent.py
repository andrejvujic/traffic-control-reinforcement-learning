from src.environment.agents.dqn.dqn import DQN
from src.environment.agents.dqn.replay_buffer import ReplayBuffer
from src.game.utilities import random_bool
from src.game.constants import TRAFFIC_LIGHT_PHASES

import random
import torch as T
import torch.nn as nn
import torch.optim as optim


INPUT_FEATURES = 40
OUTPUT_FEATURES = len(TRAFFIC_LIGHT_PHASES) + 1


class DQNAgent:
    def __init__(
        self,
        memory_size=100000,
        batch_size=64,
        alpha=0.0001,
        gamma=0.95,
        grad_norm_clipping_th=5.0
    ):
        self.policy_network = DQN(
            INPUT_FEATURES,
            OUTPUT_FEATURES
        )

        self.target_network = DQN(
            INPUT_FEATURES,
            OUTPUT_FEATURES
        )

        self.replay_buffer = ReplayBuffer(
            memory_size
        )

        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            alpha
        )

        self.batch_size = batch_size
        self.gamma = gamma
        self.grad_norm_clipping_th = grad_norm_clipping_th

        self.sync_networks()

    def sync_networks(self):
        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )

    def remember(self, state, action, new_state, reward, terminated_flag):
        self.replay_buffer.add(
            state,
            action,
            new_state,
            reward,
            terminated_flag
        )

    def next_action(self, state, epsilon=0.0):
        if random_bool(true_probability=epsilon):
            return random.choice(
                range(OUTPUT_FEATURES)
            )

        with T.no_grad():
            action = self.policy_network(
                T.tensor(state, dtype=T.float32)
            ).argmax()
            return action.item()

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        loss_function = nn.HuberLoss()
        states, actions, new_states, rewards, terminated_flags = self.replay_buffer.get_tensors(
            self.replay_buffer.sample(self.batch_size)
        )

        current_q = self.policy_network(states)

        current_q = current_q.gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        with T.no_grad():
            next_q, _ = self.target_network(new_states).max(dim=1)
            target_q = rewards + self.gamma * next_q * (1 - terminated_flags)

        loss = loss_function(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        T.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(),
            self.grad_norm_clipping_th,
        )
        self.optimizer.step()

    def save(self, path):
        T.save(
            self.policy_network.state_dict(),
            path
        )

    def load(self, path):
        self.policy_network.load_state_dict(
            T.load(path)
        )
