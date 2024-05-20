from __future__ import annotations

import math
import os

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import mujoco
import glfw
import cv2
import tools
import time


# class Agent:
#     """Agent that learns to solve the Inverted Pendulum task using a policy gradient algorithm.
#     The agent utilizes a policy network to sample actions and update its policy based on
#     collected rewards.
#     """
#
#     def __init__(self, obs_space_dims: int, action_space_dims: int):
#         """Initializes the agent with a neural network policy.
#
#         Args:
#             obs_space_dims (int): Dimension of the observation space.
#             action_space_dims (int): Dimension of the action space.
#         """
#         self.policy_network = Policy_Network(obs_space_dims, action_space_dims)
#         self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=5e-5)
#
#     def sample_action(self, state: np.ndarray) -> tuple[float, torch.Tensor]:
#         """Samples an action according to the policy network given the current state.
#
#         Args:
#             state (np.ndarray): The current state observation from the environment.
#
#         Returns:
#             tuple[float, torch.Tensor]: The action sampled from the policy distribution and its log probability.
#         """
#         state = torch.tensor(state, dtype=torch.float32)
#         mean, std = self.policy_network(state)
#         dist = Normal(mean, std)
#         action = dist.sample()
#         log_prob = dist.log_prob(action)
#         return action.item(), log_prob
#
#     def update(self, rewards, log_probs):
#         """Updates the policy network using the REINFORCE algorithm based on collected rewards and log probabilities.
#
#         Args:
#             rewards (list): Collected rewards from the environment.
#             log_probs (list): Log probabilities of the actions taken.
#         """
#         discounted_rewards = []
#         cumulative_reward = 0
#         for reward in reversed(rewards):
#             cumulative_reward = reward + 0.99 * cumulative_reward
#             discounted_rewards.insert(0, cumulative_reward)
#         discounted_rewards = torch.tensor(discounted_rewards)
#         log_probs = torch.stack(log_probs)
#
#         loss = -log_probs * discounted_rewards
#         loss = loss.sum()
#
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#
#         # @staticmethod
#         # def save_model(model: nn.Module, path: str):
#         #     """Saves the policy network to the specified path."""
#         #     torch.save(model.state_dict(), path)
#         #
#         # @staticmethod
#         # def load_model(model: nn.Module, path: str):
#         #     """Loads the policy network from the specified path."""
#         #     if os.path.exists(path):
#         #         model.load_state_dict(torch.load(path))
#         #         model.eval()
#         #     else:
#         #         print(f"No model found at {path}")
#
#
# class Policy_Network(nn.Module):
#     """Neural network to parameterize the policy by predicting action distribution parameters."""
#
#     def __init__(self, obs_space_dims: int, action_space_dims: int):
#         """Initializes layers of the neural network.
#
#         Args:
#             obs_space_dims (int): Dimension of the observation space.
#             action_space_dims (int): Dimension of the action space.
#         """
#         super().__init__()
#         self.fc1 = nn.Linear(obs_space_dims, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.mean = nn.Linear(128, action_space_dims)
#         self.std = nn.Linear(128, action_space_dims)
#
#     def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         """Predicts parameters of the action distribution given the state.
#
#         Args:
#             x (torch.Tensor): The state observation.
#
#         Returns:
#             tuple[torch.Tensor, torch.Tensor]: Predicted mean and standard deviation of the action distribution.
#         """
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         mean = self.mean(x)
#         std = torch.exp(self.std(x))
#         return mean, std
#
#
# def init_mujoco(model_path):
#     mujoco_model = mujoco.MjModel.from_xml_path(model_path)
#     mujoco_data = mujoco.MjData(mujoco_model)
#     return mujoco_model, mujoco_data


def init_glfw(width=640, height=480):
    if not glfw.init():
        return None
    window = glfw.create_window(width, height, "MuJoCo Simulation", None, None)
    if not window:
        glfw.terminate()
        return None
    glfw.make_context_current(window)
    return window


def render(window, mujoco_model, mujoco_data):
    width, height = glfw.get_framebuffer_size(window)
    scene = mujoco.MjvScene(mujoco_model, maxgeom=1000)
    context = mujoco.MjrContext(mujoco_model, mujoco.mjtFontScale.mjFONTSCALE_150)

    mujoco.mjv_updateScene(mujoco_model, mujoco_data, mujoco.MjvOption(), None, mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(mujoco.MjrRect(0, 0, width, height), scene, context)

    buffer = np.zeros((height, width, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(buffer, None, mujoco.MjrRect(0, 0, width, height), context)
    buffer = np.flipud(buffer)
    cv2.imshow("MuJoCo Simulation", buffer)
    cv2.waitKey(1)


if __name__ == "__main__":
    xml_path = "inverted_pendulum.xml"
    model_path = "policy_network.pth"

    model, data = tools.init_mujoco(xml_path)
    window = init_glfw()

    if window:
        obs_space_dims = model.nq
        action_space_dims = model.nu
        agent = tools.Agent(obs_space_dims, action_space_dims)
        tools.load_model(agent.policy_network, model_path)
        total_num_episodes = int(20) # training epochs

        for episode in range(total_num_episodes):
            rewards = []
            log_probs = []
            done = False
            data.time = 0

            # reset the state
            mujoco.mj_resetData(model, data)
            print('init angle: ', data.qpos[1])
            print('episode:', episode)
            while not done:
                state = data.qpos
                action, log_prob = agent.sample_action(state)
                data.ctrl[0] = action
                mujoco.mj_step(model, data)
                reward = - (data.qpos[1] ** 2)  # Example reward function
                rewards.append(reward)
                log_probs.append(log_prob)
                done = data.time > 2.5  # Example condition to end episode


                render(window, model, data)

            agent.update(rewards, log_probs)

        glfw.terminate()
        tools.save_model(agent.policy_network, model_path)
