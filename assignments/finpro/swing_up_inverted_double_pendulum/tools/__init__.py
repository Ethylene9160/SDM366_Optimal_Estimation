from __future__ import annotations

import os
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import mujoco
import mujoco.viewer


from . import DDPG

from .DDPG import DDPGAgent
from .A2C import A2CAgent
from .Q import QLearningAgent
from .DQN import DQNAgent


np.random.seed(0)

def init_mujoco(model_path):
    mujoco_model = mujoco.MjModel.from_xml_path(model_path)
    mujoco_data = mujoco.MjData(mujoco_model)
    return mujoco_model, mujoco_data

def random_state(data):
    x = np.random.uniform(-0.1, 0.1)
    theta1 = np.random.uniform(-0.2, 0.2)
    theta2 = np.random.uniform(-0.2, 0.2)
    dx = np.random.uniform(-1.1, 1.1)
    dtheta1 = np.random.uniform(-0.8, 0.8)
    dtheta2 = np.random.uniform(-1.1, 1.1)

def get_obs(data):
    '''
    obs[0]: position of cart
    obs[1]: theta
    obs[2]: v of the cart
    obs[3]: omega
    obs[4]: sin(angle)
    obs[5]: cos(angle)
    '''
    theta = data.qpos[1:]
    theta = np.where(theta >= np.pi, theta - np.pi, theta)
    theta = np.where(theta < -np.pi, theta + np.pi, theta)
    return np.concatenate(
        [
            data.qpos[:1],
            theta,
            data.qvel,
            # np.sin(data.qpos[1:]),
            # np.cos(data.qpos[1:])
        ]
    ).ravel()

def f(xk, uk):
    '''
    xk[0]: x of the cart
    xk[1]: angle 1
    xk[2]: angle between the first and the second
    xk[3]: dx/dt
    xk[4]: dtheta1/dt
    sk[5]: dtheta2/dt
    '''
    xk_star = 0
    yk_star = 0
    return xk_star, yk_star