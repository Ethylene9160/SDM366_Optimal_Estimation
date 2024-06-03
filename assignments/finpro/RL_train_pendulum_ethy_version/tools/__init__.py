from __future__ import annotations

import os
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import mujoco
import mujoco.viewer
def save_model(model: nn.Module, path: str):
    """Saves the policy network to the specified path."""
    torch.save(model.state_dict(), path)
    print(f"Model saved at {path}")


def load_model(model: nn.Module, path: str):
    """Loads the policy network from the specified path."""
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        model.eval()
    else:
        print(f"No model found at {path}")


class Agent:
    """Agent that learns to solve the Inverted Pendulum task using a policy gradient algorithm.
    The agent utilizes a policy network to sample actions and update its policy based on
    collected rewards.
    """

    def __init__(self, obs_space_dims: int, action_space_dims: int, lr: float = 1e-3, gama:float = 0.95):
        """Initializes the agent with a neural network policy.

        Args:
            obs_space_dims (int): Dimension of the observation space.
            action_space_dims (int): Dimension of the action space.
            lr (float): Learning rate for the optimizer.
        """
        self.policy_network = Policy_Network(obs_space_dims, action_space_dims)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.gamma = gama  # Discount factor

    def sample_action(self, state: np.ndarray) -> tuple[float, torch.Tensor]:
        """Samples an action according to the policy network given the current state.

        Args:
            state (np.ndarray): The current state observation from the environment.

        Returns:
            tuple[float, torch.Tensor]: The action sampled from the policy distribution and its log probability.
        """
        state = torch.FloatTensor(state)
        mean, std = self.policy_network(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def update(self, rewards: list, log_probs: list):
        """Updates the policy network using the REINFORCE algorithm based on collected rewards and log probabilities.

        Args:
            rewards (list): Collected rewards from the environment.
            log_probs (list): Log probabilities of the actions taken.
        """
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)

        discounted_rewards = torch.FloatTensor(discounted_rewards)
        log_probs = torch.stack(log_probs)

        loss = -torch.sum(log_probs * discounted_rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Policy_Network(nn.Module):
    """Neural network to parameterize the policy by predicting action distribution parameters."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes layers of the neural network.

        Args:
            obs_space_dims (int): Dimension of the observation space.
            action_space_dims (int): Dimension of the action space.
        """
        super().__init__()
        self.fc1 = nn.Linear(obs_space_dims, 128)
        self.fc2 = nn.Linear(128, 64)
        self.mean = nn.Linear(64, action_space_dims)
        self.log_std = nn.Linear(64, action_space_dims)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predicts parameters of the action distribution given the state.

        Args:
            x (torch.Tensor): The state observation.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Predicted mean and standard deviation of the action distribution.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        std = torch.exp(log_std)
        return mean, std


def init_mujoco(model_path):
    mujoco_model = mujoco.MjModel.from_xml_path(model_path)
    mujoco_data = mujoco.MjData(mujoco_model)
    return mujoco_model, mujoco_data