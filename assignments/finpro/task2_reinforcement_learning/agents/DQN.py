import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import mujoco
import mujoco.viewer
import time


class DQNAgent:
    """Agent that learns to solve the Inverted Pendulum task using DQN."""

    def __init__(self, obs_space_dims: int, action_space: list[float], alpha: float = 0.001, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995, epsilon_min: float = 0.01):
        """Initializes the agent with a neural network policy and hyperparameters.

        Args:
            obs_space_dims (int): Dimension of the observation space.
            action_space (list[float]): List of possible actions.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Exploration rate.
            epsilon_decay (float): Decay rate for exploration probability.
            epsilon_min (float): Minimum exploration probability.
        """
        self.obs_space_dims = obs_space_dims
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = deque(maxlen=2000)
        self.batch_size = 64

        self.policy_network = self.build_model()
        self.target_network = self.build_model()
        self.update_target_network()
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.alpha)

    def build_model(self) -> nn.Module:
        """Builds the neural network model for approximating Q-values.

        Returns:
            nn.Module: The neural network model.
        """
        model = nn.Sequential(
            nn.Linear(self.obs_space_dims, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, len(self.action_space))
        )
        return model

    def update_target_network(self):
        """Updates the target network weights with the policy network weights."""
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Stores experience in the replay memory.

        Args:
            state (np.ndarray): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (np.ndarray): The next state observed.
            done (bool): Whether the episode is finished.
        """
        self.memory.append((state, action, reward, next_state, done))

    def sample_action(self, state: np.ndarray) -> int:
        """Samples an action using an epsilon-greedy policy.

        Args:
            state (np.ndarray): The current state observation from the environment.

        Returns:
            int: The action index selected from the action space.
        """
        if np.random.rand() <= self.epsilon:
            return random.choice(range(len(self.action_space)))
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_network(state_tensor)
            return torch.argmax(q_values, dim=1).item()

    def replay(self):
        """Performs experience replay to train the policy network."""
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.target_network(next_state_tensor)).item()
            target_f = self.policy_network(state_tensor).detach().numpy()
            target_f[0][action] = target
            target_tensor = torch.FloatTensor(target_f)

            self.optimizer.zero_grad()
            output = self.policy_network(state_tensor)
            loss = nn.MSELoss()(output, target_tensor)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, path: str):
        """Saves the policy network to a file.

        Args:
            path (str): The path to save the policy network.
        """
        torch.save(self.policy_network.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Loads the policy network from a file.

        Args:
            path (str): The path to load the policy network from.
        """
        self.policy_network.load_state_dict(torch.load(path))
        self.update_target_network()
        print(f"Model loaded from {path}")