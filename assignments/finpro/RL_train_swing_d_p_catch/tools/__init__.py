from __future__ import annotations

import os

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from . import A2C
from . import DDPG


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


class CustomEnv(gym.Env):
    def __init__(self, model, data, max_time=120):
        super(CustomEnv, self).__init__()
        self.model = model
        self.data = data
        self.max_time = max_time
        self.episode_reward = 0

        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(model.nu,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

    def step(self, action):
        self.data.ctrl[0] = action
        mujoco.mj_step(self.model, self.data)
        state = self._get_obs()

        # reward = self.reward_function(state, action)
        # p = 0.85 * self.data.qpos[0] ** 2 + 0.005 * self.data.qvel[0] ** 2 + 0.05 * self.data.qvel[1] ** 2
        # reward = -p

        x, _, y = self.data.site_xpos[0]
        dist_penalty = 0.08 * x ** 2 + 10.0 * (y - 2) ** 2
        v1, v2 = self.data.qvel[1:3]
        vel_penalty = 8e-3 * v1 ** 2 + 4e-2 * v2 ** 2
        reward = 10 - dist_penalty - vel_penalty

        # reward = 1.0
        reward = float(reward)

        # Accumulate the reward
        self.episode_reward += reward

        theta_1 = np.arctan2(state[6], state[8])
        theta_2 = np.arctan2(state[7], state[9])
        theta = theta_1 + theta_2
        # done = bool(self.data.time > self.max_time or abs(theta) > 0.75)
        done = bool(self.data.time > self.max_time or y < 0.99)
        truncated = bool(self.data.time > self.max_time)
        # if done:
        #     print(f"Episode ended at time {self.data.time}, x = {state[0]}", reward)
        return state, reward, done, truncated, {}

    def reset(self, seed=None):
        """Randomize the high state."""
        np.random.seed(seed)
        mujoco.mj_resetData(self.model, self.data)
        random_state(self.data, seed)
        self.episode_reward = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return get_obs_lifer(self.data).astype(np.float32)

    def render(self, mode='human'):
        pass

    def close(self):
        pass


def init_mujoco(model_path):
    mujoco_model = mujoco.MjModel.from_xml_path(model_path)
    mujoco_data = mujoco.MjData(mujoco_model)
    return mujoco_model, mujoco_data


def random_state(data, seed=2333):
    np.random.seed(seed)
    init_x = np.random.uniform(-0.75, 0.75)
    init_theta1 = np.random.uniform(-0.2, 0.2)
    init_theta2 = np.random.uniform(-0.2, 0.2)
    init_v = np.random.uniform(-1.0, 1.0)
    init_omega1 = np.random.uniform(-1.0, 1.0)
    init_omega2 = np.random.uniform(-1.0, 1.0)
    data.qpos[0] = init_x
    data.qpos[1] = init_theta1
    data.qpos[2] = init_theta2
    data.qvel[0] = init_v
    data.qvel[1] = init_omega1
    data.qvel[2] = init_omega2
    # print(f"Initial x: {init_x}, theta1: {init_theta1}, theta2: {init_theta2}")


def get_obs_lifer(data):
    """获取环境的观测状态."""
    state = np.concatenate(
        [
            data.qpos,
            data.qvel,
            np.sin(data.qpos[1:]),
            np.cos(data.qpos[1:])
        ]
    ).ravel()
    # state = [x, theta 1, theta 2, v, omega 1, omega 2, sin(theta 1), sin(theta 2), cos(theta 1), cos(theta 2)]
    return state
