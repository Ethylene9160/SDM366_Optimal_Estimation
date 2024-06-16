import os
import sys

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from util import init_mujoco
from agents import A2CAgent,  DDPGAgent, PILCOAgent, DQNAgent, QAgent

def random_state(data):
    init_x = np.random.uniform(-0.1, 0.1)
    init_theta1 = np.random.uniform(-0.1, 0.1)
    init_theta2 = np.random.uniform(-0.1, 0.1)
    init_v = np.random.uniform(-0.1, 0.1)
    init_omega1 = np.random.uniform(-0.1, 0.1)
    init_omega2 = np.random.uniform(-0.1, 0.1)
    data.qpos[0] = init_x
    data.qpos[1] = init_theta1
    data.qpos[2] = init_theta2
    data.qvel[0] = init_v
    data.qvel[1] = init_omega1
    data.qvel[2] = init_omega2

def get_obs(data):
    return np.concatenate(
        [
            data.qpos,
            data.qvel,
            np.sin(data.qpos[1:]),
            np.cos(data.qpos[1:])
        ]
    ).ravel()