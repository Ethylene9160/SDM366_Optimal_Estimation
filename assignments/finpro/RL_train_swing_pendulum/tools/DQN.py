import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class DQNAgent:
    """Agent that learns to solve the Inverted Pendulum task using a DQN algorithm."""

    def __init__(self, obs_space_dims: int, action_space_dims: int, lr: float = 1e-3, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995, epsilon_min: float = 0.01):
        """Initializes the agent with a Q-network and target Q-network.

        Args:
            obs_space_dims (int): Dimension of the observation space.
            action_space_dims (int): Dimension of the action space.
            lr (float): Learning rate for the optimizer.
            gamma (float): Discount factor.
            epsilon (float): Initial exploration rate.
            epsilon_decay (float): Decay rate for exploration.
            epsilon_min (float): Minimum exploration rate.
        """
        self.q_network = QNetwork(obs_space_dims, action_space_dims)
        self.target_network = QNetwork(obs_space_dims, action_space_dims)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 64
        self.update_target_network()

    def update_target_network(self):
        """Updates the target network to match the Q-network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def sample_action(self, state: np.ndarray) -> float:
        """Samples an action using ε-greedy policy given the current state.

        Args:
            state (np.ndarray): The current state observation from the environment.
            action_space (gym.spaces.Box): The action space of the environment.

        Returns:
            float: The action sampled from the policy distribution.
        """
        if random.random() < self.epsilon:
            # AttributeError: 'numpy.ndarray' object has no attribute 'low'
            # return random.uniform(action_space.low[0], action_space.high[0])
            return random.uniform(-5, 5)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state)
            return q_values.detach().numpy()[0, 0]

    def store_transition(self, state, action, reward, next_state, done):
        """Stores a transition in the replay buffer.

        Args:
            state (np.ndarray): The current state observation from the environment.
            action (float): The action taken.
            reward (float): The reward received.
            next_state (np.ndarray): The next state observation.
            done (bool): Whether the episode is done.
        """
        self.replay_buffer.append((state, action, reward, next_state, done))

    def update(self):
        """Updates the Q-network using samples from the replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 优化列表转换为 Tensor
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions)).unsqueeze(1)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)

        # 计算当前 Q 值
        current_q_values = self.q_network(states)

        # 计算目标 Q 值
        next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # 计算损失
        loss = nn.MSELoss()(current_q_values, target_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, path: str):
        """Saves the Q-network and target network to the specified path.

        Args:
            path (str): The path where the model will be saved.
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        print(f'Successfully saved model to {path}.')

    def load_model(self, path: str):
        """Loads the Q-network and target network from the specified path.

        Args:
            path (str): The path from where the model will be loaded.
        """
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Successfully loaded model from {path}.")


class QNetwork(nn.Module):
    """Neural network to parameterize the Q-function."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes layers of the neural network.

        Args:
            obs_space_dims (int): Dimension of the observation space.
            action_space_dims (int): Dimension of the action space.
        """
        super().__init__()
        self.fc1 = nn.Linear(obs_space_dims, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_space_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predicts Q-values given the state.

        Args:
            x (torch.Tensor): The state observation.

        Returns:
            torch.Tensor: Predicted Q-values for each action.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values
