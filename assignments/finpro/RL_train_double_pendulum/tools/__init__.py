from __future__ import annotations

import os
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import mujoco
import mujoco.viewer

from . import Agent
from . import DQN, A3C

from .A3C import A3CAgent

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