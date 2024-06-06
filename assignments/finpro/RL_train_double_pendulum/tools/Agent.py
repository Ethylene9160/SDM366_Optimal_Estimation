from __future__ import annotations

import os
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import mujoco
import mujoco.viewer

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers):
        super(ActorCritic, self).__init__()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(state_dim, hidden_layers[0]))
        for i in range(1, len(hidden_layers)):
            self.fcs.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
        self.actor = nn.Linear(hidden_layers[-1], action_dim)
        self.critic = nn.Linear(hidden_layers[-1], 1)

    def forward(self, x):
        for i in range(len(self.fcs)):
            x = torch.relu(self.fcs[i](x))
        # print(self.actor(x))
        return torch.softmax(self.actor(x), dim=0), self.critic(x)

class Agent(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, lr=0.001, gamma=0.99, device = 'cpu'):
        super(Agent, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ELU(x)
        x = self.fc2(x)
        x = torch.softmax(x, dim=1)
        return x

    def update(self, rewards: list, log_probs: list):
        """Updates the policy network based on the collected rewards."""
        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(rewards))):
            G = rewards[i] + self.gamma * G
            action_log_prob = torch.log()

