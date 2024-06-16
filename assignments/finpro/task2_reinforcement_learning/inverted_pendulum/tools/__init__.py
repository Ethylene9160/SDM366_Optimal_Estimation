import os
import sys

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from util import init_mujoco
from agents import A2CAgent,  DDPGAgent, PILCOAgent, DQNAgent, QAgent

def random_state(data):
    init_x = np.random.uniform(-0.1, 0.1)
    init_theta = np.random.uniform(-0.1, 0.1)
    init_v = np.random.uniform(-0.1, 0.1)
    init_omega = np.random.uniform(-0.1, 0.1)
    data.qpos[0] = init_x
    data.qpos[1] = init_theta
    data.qvel[0] = init_v
    data.qvel[1] = init_omega

def hard_mode_random_state(data):
    pass

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