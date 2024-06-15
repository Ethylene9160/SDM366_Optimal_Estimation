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

def random_state(data):
    x = np.random.uniform(-0.1, 0.1)
    theta1 = np.random.uniform(-0.2, 0.2)
    theta2 = np.random.uniform(-0.2, 0.2)
    dx = np.random.uniform(-0.3, 0.3)
    dtheta1 = np.random.uniform(-0.2, 0.2)
    dtheta2 = np.random.uniform(-0.2, 0.2)

def large_random(data):
    x = np.random.uniform(-1, 1)
    theta1 = np.random.uniform(-np.pi, np.pi)
    theta2 = np.random.uniform(-np.pi, np.pi)
    dx = np.random.uniform(-2, 2)
    dtheta1 = np.random.uniform(-3, 3)
    dtheta2 = np.random.uniform(-3, 3)
    data.qpos[0] = x
    data.qpos[1] = theta1
    data.qpos[2] = theta2
    data.qvel[0] = dx
    data.qvel[1] = dtheta1
    data.qvel[2] = dtheta2


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

def get10obs(data):
    # return np.concatenate(
    #     [
    #         data.qpos[:1],  # cart x pos
    #         np.sin(data.qpos[1:]),  # link angles
    #         np.cos(data.qpos[1:]),
    #         np.clip(data.qvel, -10, 10),
    #         np.clip(data.qfrc_constraint, -10, 10),
    #     ]
    # ).ravel()
    theta = data.qpos[1:]
    theta = np.where(theta >= np.pi, theta - np.pi, theta)
    theta = np.where(theta < -np.pi, theta + np.pi, theta)
    return np.concatenate(
        [
            data.qpos[:1],
            theta,
            data.qvel,
            np.sin(data.qpos[1:]),
            np.cos(data.qpos[1:])
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

# C = np.array([
#     [1,-0.5,0,-0.5,0],
#     [0,0,0.5,0,0.5]])
#
# print(C.T@C)