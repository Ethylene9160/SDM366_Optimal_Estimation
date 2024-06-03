import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

class A3CAgent:
    """Agent that learns to solve the Inverted Pendulum task using an Actor-Critic algorithm.
    The agent utilizes a policy network to sample actions and a value network to estimate state values.
    """

    def __init__(self, obs_space_dims: int, action_space_dims: int, lr: float = 1e-3, gamma: float = 0.95):
        """Initializes the agent with a neural network policy and value network.

        Args:
            obs_space_dims (int): Dimension of the observation space.
            action_space_dims (int): Dimension of the action space.
            lr (float): Learning rate for the optimizer.
        """
        self.policy_network = PolicyNetwork(obs_space_dims, action_space_dims)
        self.value_network = ValueNetwork(obs_space_dims)
        self.optimizer_policy = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.optimizer_value = optim.Adam(self.value_network.parameters(), lr=lr)
        self.gamma = gamma  # Discount factor

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

    def update(self, rewards: list, log_probs: list, states: list):
        """Updates the policy network and value network using the Actor-Critic algorithm.

        Args:
            rewards (list): Collected rewards from the environment.
            log_probs (list): Log probabilities of the actions taken.
            states (list): States encountered in the episode.
        """
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)

        discounted_rewards = torch.FloatTensor(discounted_rewards)
        log_probs = torch.stack(log_probs)

        states = torch.FloatTensor(states)
        values = self.value_network(states)

        # Ensure the shapes are compatible for MSELoss
        values = values.squeeze()  # Shape: [batch_size]
        advantages = discounted_rewards - values

        # Update policy network
        policy_loss = -torch.sum(log_probs * advantages.detach())
        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        self.optimizer_policy.step()

        # Update value network
        value_loss = nn.MSELoss()(values, discounted_rewards)
        self.optimizer_value.zero_grad()
        value_loss.backward()
        self.optimizer_value.step()

    def save_model(self, path: str):
        """Saves the policy and value networks to the specified path.

        Args:
            path (str): The path where the model will be saved.
        """
        torch.save({
            'policy_network_state_dict': self.policy_network.state_dict(),
            'value_network_state_dict': self.value_network.state_dict(),
            'optimizer_policy_state_dict': self.optimizer_policy.state_dict(),
            'optimizer_value_state_dict': self.optimizer_value.state_dict()
        }, path)
        print(f'successfully save model to {path}.')

    def load_model(self, path: str):
        """Loads the policy and value networks from the specified path.

        Args:
            path (str): The path from where the model will be loaded.
        """
        checkpoint = torch.load(path)
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.value_network.load_state_dict(checkpoint['value_network_state_dict'])
        self.optimizer_policy.load_state_dict(checkpoint['optimizer_policy_state_dict'])
        self.optimizer_value.load_state_dict(checkpoint['optimizer_value_state_dict'])
        self.policy_network.train()
        self.value_network.train()
        print(f"successfully load model from {path}.")


class PolicyNetwork(nn.Module):
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


class ValueNetwork(nn.Module):
    """Neural network to estimate the value of a given state."""

    def __init__(self, obs_space_dims: int):
        """Initializes layers of the neural network.

        Args:
            obs_space_dims (int): Dimension of the observation space.
        """
        super().__init__()
        self.fc1 = nn.Linear(obs_space_dims, 128)
        self.fc2 = nn.Linear(128, 64)
        self.value = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predicts the value of the given state.

        Args:
            x (torch.Tensor): The state observation.

        Returns:
            torch.Tensor: Predicted value of the state.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.value(x)
        return value
