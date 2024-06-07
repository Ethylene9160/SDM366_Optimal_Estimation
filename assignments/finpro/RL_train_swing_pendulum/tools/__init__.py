from __future__ import annotations

import os
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import mujoco
import mujoco.viewer

from . import A2C
from . import DQN

from .A2C import A2CAgent
from .DQN import DQNAgent


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


np.random.seed(0)


def init_mujoco(model_path):
    mujoco_model = mujoco.MjModel.from_xml_path(model_path)
    mujoco_data = mujoco.MjData(mujoco_model)
    return mujoco_model, mujoco_data


def random_state(data):
    init_x = np.random.uniform(-0.1, 0.1)
    init_theta = np.random.uniform(-0.1, 0.1)
    init_v = np.random.uniform(-0.1, 0.1)
    init_omega = np.random.uniform(-0.1, 0.1)
    data.qpos[0] = init_x
    data.qpos[1] = init_theta
    data.qvel[0] = init_v
    data.qvel[1] = init_omega


def get_obs(data):
    # return np.concatenate(
    #     [
    #         data.qpos[:1],  # cart x pos
    #         np.sin(data.qpos[1:]),  # link angles
    #         np.cos(data.qpos[1:]),
    #         np.clip(data.qvel, -10, 10),
    #         np.clip(data.qfrc_constraint, -10, 10),
    #     ]
    # ).ravel()
    return np.concatenate(
        [
            data.qpos,
            data.qvel,
            np.sin(data.qpos[1:]),
            np.cos(data.qpos[1:])
        ]
    ).ravel()

def get_obs_lifer(data):
    """获取环境的观测状态."""
    x = data.qpos[0]  # 小车的位置
    theta = data.qpos[1]  # 第一根杆的角度
    sin_theta = np.sin(data.qpos[1])  # 第一根杆的角度的正弦值
    cos_theta = np.cos(data.qpos[1])  # 第一根杆的角度的余弦值
    x_dot = data.qvel[0]  # 小车的速度
    theta_dot = data.qvel[1]  # 第一根杆的角速度
    return np.array([x, theta, sin_theta, cos_theta, x_dot, theta_dot])
