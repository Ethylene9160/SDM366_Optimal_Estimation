import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

class A2CAgent:
    def __init__(self, obs_space_dims: int, action_space_dims: int, lr: float = 1e-3, gamma: float = 0.95, device='cpu'):
        self.device = torch.device(device)
        self.policy_network = PolicyNetwork(obs_space_dims, action_space_dims).to(self.device)
        self.value_network = ValueNetwork(obs_space_dims).to(self.device)
        self.optimizer_policy = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.optimizer_value = optim.Adam(self.value_network.parameters(), lr=lr)
        self.gamma = gamma

    def sample_action(self, state: np.ndarray) -> tuple[float, torch.Tensor]:
        state = torch.FloatTensor(state).to(self.device)
        mean, std = self.policy_network(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()  # Ensure log_prob is a scalar
        return action.item(), log_prob

    def update(self, rewards: list, log_probs: list, states: list):
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)

        discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.device)
        log_probs = torch.stack(log_probs).to(self.device)
        states = np.array(states)
        states = torch.FloatTensor(states).to(self.device)
        values = self.value_network(states).squeeze()

        advantages = discounted_rewards - values

        policy_loss = -torch.sum(log_probs * advantages.detach())
        self.optimizer_policy.zero_grad()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1)
        policy_loss.backward()
        self.optimizer_policy.step()

        value_loss = nn.MSELoss()(values, discounted_rewards)
        self.optimizer_value.zero_grad()
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 1)
        value_loss.backward()
        self.optimizer_value.step()

    def save_model(self, path: str):
        torch.save({
            'policy_network_state_dict': self.policy_network.state_dict(),
            'value_network_state_dict': self.value_network.state_dict(),
            'optimizer_policy_state_dict': self.optimizer_policy.state_dict(),
            'optimizer_value_state_dict': self.optimizer_value.state_dict()
        }, path)
        print(f'successfully save model to {path}.')

    def load_model(self, path: str):
        checkpoint = torch.load(path)
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.value_network.load_state_dict(checkpoint['value_network_state_dict'])
        self.optimizer_policy.load_state_dict(checkpoint['optimizer_policy_state_dict'])
        self.optimizer_value.load_state_dict(checkpoint['optimizer_value_state_dict'])
        self.policy_network.train()
        self.value_network.train()
        print(f"successfully load model from {path}.")

class PolicyNetwork(nn.Module):
    def __init__(self, obs_space_dims: int, action_space_dims: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_space_dims, 128)
        self.fc2 = nn.Linear(128, 64)
        self.mean = nn.Linear(64, action_space_dims)
        self.log_std = nn.Linear(64, action_space_dims)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(min=-20, max=2)  # Clamping the log_std to avoid numerical instability
        std = torch.exp(log_std)
        return mean, std

class ValueNetwork(nn.Module):
    def __init__(self, obs_space_dims: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_space_dims, 128)
        self.fc2 = nn.Linear(128, 64)
        self.value = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.value(x)
        return value
