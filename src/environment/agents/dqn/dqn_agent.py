from src.environment.agents.dqn.network import Network
from src.environment.agents.dqn.replay_buffer import ReplayBuffer
from src.game.utilities import random_bool
from src.game.constants import TRAFFIC_LIGHT_PHASES
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
        memory_size=4096,
        batch_size=32,
        batch_count=1,
        alpha=0.0002,
        gamma=0.9,
        epochs=1
    ):
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

        with T.no_grad():
            actions = self.policy_network(
                T.tensor(state, dtype=T.float32)
            )
            return actions.argmax().item(), False

    def learn(self):
        if len(self.replay_buffer) < self.batch_size * self.batch_count:
            return

        loss_function = nn.SmoothL1Loss()
        for states, actions, new_states, rewards, terminated_flags in islice(
            self.replay_buffer.sample(), self.batch_count
        ):
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

    ACTIONS = range(len(TRAFFIC_LIGHT_PHASES) + 1)
