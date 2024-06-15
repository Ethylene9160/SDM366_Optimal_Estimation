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
from .PILCO import PILCOAgent, current_reward


np.random.seed(0)

def init_mujoco(model_path):
    mujoco_model = mujoco.MjModel.from_xml_path(model_path)
    mujoco_data = mujoco.MjData(mujoco_model)
    return mujoco_model, mujoco_data

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
    if theta >= np.pi:
        theta -= 2*np.pi
    elif theta < -np.pi:
        theta += 2*np.pi
    return np.concatenate(
        [
            data.qpos[:1],
            [theta],
            data.qvel,
        ]
    ).ravel()

def get6obs(data):
    '''
        obs[0]: position of cart
        obs[1]: theta
        obs[2]: v of the cart
        obs[3]: omega
        obs[4]: sin(angle)
        obs[5]: cos(angle)
        '''
    theta = data.qpos[1]
    if theta >= np.pi:
        theta -= 2 * np.pi
    elif theta < -np.pi:
        theta += 2 * np.pi
    return np.concatenate(
        [
            data.qpos[:1],
            [theta],
            data.qvel,
            np.sin(data.qpos[1:]),
            np.cos(data.qpos[1:])
        ]
    ).ravel()
def chaotic_state(data):
    x = np.random.uniform(-1,1)
    dx = np.random.uniform(-5,5)
    theta = np.random.uniform(-np.pi, np.pi-1e-5)
    dtheta = np.random.uniform(-8,8)


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