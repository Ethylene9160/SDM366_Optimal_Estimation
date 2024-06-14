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


class CustomEnv_a(gym.Env):
    def __init__(self, model, data, reward_function, max_time=120, theta_threshold=0.5):
        super(CustomEnv_a, self).__init__()
        self.model = model
        self.data = data
        self.reward_function = reward_function
        self.max_time = max_time
        self.theta_threshold = theta_threshold

        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(model.nu,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

    def step(self, action):
        self.data.ctrl[0] = action * 3.0
        mujoco.mj_step(self.model, self.data)
        state = self._get_obs()

        # reward = self.reward_function(state, action)
        reward = 1.0
        reward = float(reward)

        theta = np.arctan2(state[2], state[3])
        done = bool(self.data.time > self.max_time or abs(theta) > self.theta_threshold)
        truncated = bool(self.data.time > self.max_time)
        # if done:
        #     print(f"Episode ended at time {self.data.time}, x = {state[0]}", reward)
        return state, reward, done, truncated, {}

    def reset(self, seed=None):
        """Randomize the high state."""
        np.random.seed(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = np.random.uniform(-1.25, 1.25)
        self.data.qpos[1] = np.random.uniform(-0.30, 0.30)
        self.data.qvel[0] = np.random.uniform(-2.0, 2.0)
        self.data.qvel[1] = np.random.uniform(-2.0, 2.0)
        return self._get_obs(), {}

    def _get_obs(self):
        # return get_obs(self.data) as float32
        # return get_obs_lifer(self.data)
        return get_obs_lifer(self.data).astype(np.float32)

    def render(self, mode='human'):
        pass

    def close(self):
        pass


class CustomEnv_b(gym.Env):
    def __init__(self, model, data, reward_function, max_time=120):
        super(CustomEnv_b, self).__init__()
        self.model = model
        self.data = data
        self.reward_function = reward_function
        self.max_time = max_time

        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(model.nu,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

    def step(self, action):
        self.data.ctrl[0] = action * 3.0
        mujoco.mj_step(self.model, self.data)
        state = self._get_obs()
        reward = self.reward_function(state, action)
        reward = float(reward)
        done = bool(self.data.time > self.max_time)  # or bool(abs(state[0]) > 1.30)
        truncated = bool(self.data.time > self.max_time)
        # if done:
        #     print(f"Episode ended at time {self.data.time}, x = {state[0]}", reward)
        return state, reward, done, truncated, {}

    def reset(self, seed=None):
        mujoco.mj_resetData(self.model, self.data)
        random_state(self.data, seed)
        # self.data.qpos[1] -= np.pi
        return self._get_obs(), {}

    def _get_obs(self):
        # return get_obs(self.data) as float32
        # return get_obs_lifer(self.data)
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
    init_x = np.random.uniform(-0.1, 0.1)
    init_theta = np.random.uniform(-0.1, 0.1) - np.pi
    init_v = np.random.uniform(-0.01, 0.01)
    init_omega = np.random.uniform(-0.01, 0.01)
    data.qpos[0] = init_x
    data.qpos[1] = init_theta
    data.qvel[0] = init_v
    data.qvel[1] = init_omega


def get_obs_lifer(data):
    """获取环境的观测状态."""
    x = data.qpos[0]  # 小车的位置
    theta = data.qpos[1]  # 第一根杆的角度
    sin_theta = np.sin(data.qpos[1])  # 第一根杆的角度的正弦值
    cos_theta = np.cos(data.qpos[1])  # 第一根杆的角度的余弦值
    x_dot = data.qvel[0]  # 小车的速度
    theta_dot = data.qvel[1]  # 第一根杆的角速度
    return np.array([x, theta, sin_theta, cos_theta, x_dot, theta_dot])
