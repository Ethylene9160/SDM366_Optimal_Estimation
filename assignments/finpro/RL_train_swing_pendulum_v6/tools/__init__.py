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

from .DDPG import DDPGAgent, ReplayBuffer
from .A2C import A2CAgent
from .Q import QLearningAgent
from .DQN import DQNAgent


np.random.seed(0)

def init_mujoco(model_path):
    mujoco_model = mujoco.MjModel.from_xml_path(model_path)
    mujoco_data = mujoco.MjData(mujoco_model)
    return mujoco_model, mujoco_data

def random_state(data):
    init_x = np.random.uniform(-0.52, 0.52)
    init_theta = np.random.uniform(-0.26, 0.26)
    init_v = np.random.uniform(-2.60, 2.60)
    init_omega = np.random.uniform(-2.55, 2.55)
    data.qpos[0] = init_x
    data.qpos[1] = init_theta
    data.qvel[0] = init_v
    data.qvel[1] = init_omega

def get_obs(data):
    '''
    obs[0]: position of cart
    obs[1]: theta
    obs[2]: v of the cart
    obs[3]: omega
    obs[4]: sin(angle)
    obs[5]: cos(angle)
    '''
    theta = data.qpos[1]
    if theta > np.pi:
        theta -= 2*np.pi
    elif theta < -np.pi:
        theta += 2*np.pi
    return np.concatenate(
        [
            data.qpos[:1],
            [theta],
            data.qvel,
            np.sin(data.qpos[1:]),
            np.cos(data.qpos[1:])
        ]
    ).ravel()