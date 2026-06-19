from src.environment.agents.dqn.network import Network
from src.environment.agents.dqn.replay_buffer import ReplayBuffer
from src.game.utilities import random_bool
from src.game.constants import TOTAL_LANES
from collections import deque
from itertools import islice

import random
import torch as T
import torch.nn as nn
import torch.optim as optim


class DQNAgent:
    def __init__(
        self,
        in_features,
        out_features,
        memory_size=2048,
        batch_size=64,
        batch_count=4,
        random_state=42,
        alpha=0.0003,
        gamma=0.9,
        epochs=1
    ):
        if random_state:
            T.manual_seed(random_state)
            random.seed(random_state)

        self.policy_network = Network(
            in_features,
            out_features
        )

        self.target_network = Network(
            in_features,
            out_features
        )

        self.replay_buffer = ReplayBuffer(
            memory_size,
            batch_size,
        )

        self.memory = deque([], memory_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), alpha)

        self.epochs = epochs
        self.batch_size = batch_size
        self.batch_count = batch_count
        self.gamma = gamma

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
            return random.choice(self.ACTIONS), True

        actions = self.policy_network(
            T.tensor(state, dtype=T.float32)
        )

        return actions.argmax().item(), False

    def learn(self):
        loss_function = nn.SmoothL1Loss()

        for _ in range(self.epochs):
            for states, actions, new_states, rewards, terminated_flags in islice(
                self.replay_buffer.sample(), self.batch_count
            ):
                current_q = self.policy_network(states)

                new_q = self.policy_network(states).clone().detach()
                with T.no_grad():
                    next_q = self.target_network(new_states)
                    next_q, _ = next_q.max(dim=1)
                    next_q = rewards + self.gamma * next_q * (1 - terminated_flags)

                for index in range(current_q.shape[0]):
                    new_q[index][actions[index].item()] = next_q[index]

                loss = loss_function(current_q, new_q)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def save(self):
        T.save(
            self.policy_network.state_dict(),
            'dqn.pt'
        )

    def load(self):
        self.policy_network.load_state_dict(
            T.load('dqn.pt')
        )

    ACTIONS = range(TOTAL_LANES)
