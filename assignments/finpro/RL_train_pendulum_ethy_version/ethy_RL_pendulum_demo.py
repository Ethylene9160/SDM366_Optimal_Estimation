from __future__ import annotations

import os

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import mujoco
import mujoco.viewer
# import glfw
import time

import tools


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
#         self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=1e-3)
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



def init_mujoco(model_path):
    mujoco_model = mujoco.MjModel.from_xml_path(model_path)
    mujoco_data = mujoco.MjData(mujoco_model)
    return mujoco_model, mujoco_data


def cal_dis(data):
    x = data.qpos[0] + 0.6*np.sin(data.qpos[1])
    y = 0.6*np.cos(data.qpos[1])
    return x**2+(y-0.6)**2

if __name__ == "__main__":
    xml_path = "inverted_pendulum.xml"

    model, data = init_mujoco(xml_path)

    obs_space_dims = model.nq
    action_space_dims = model.nu
    agent = tools.Agent(obs_space_dims, action_space_dims)

    model_path = "models/official_model0.pth"
    tools.load_model(agent.policy_network, model_path)
    # create viewer
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        total_num_episodes = int(3e3)
        episode = 0

        while viewer.is_running() and episode < total_num_episodes:
            rewards = []
            log_probs = []
            done = False
            data.time = 0
            print('the episode is:', episode)

            # 重置环境到初始状态
            mujoco.mj_resetData(model, data)

            while not done:
                step_start = time.time()
                state = data.qpos
                action, log_prob = agent.sample_action(state)
                data.ctrl[0] = action
                mujoco.mj_step(model, data)

                posfactor = 1.0
                if abs(data.qpos[0]) > 0.7:
                    posfactor = 5000.0
                elif abs(data.qpos[0]) > 0.6:
                    posfactor = 600.0
                elif abs(data.qpos[0]) > 0.5:
                    posfactor = 100.0
                elif abs(data.qpos[0]) > 0.4:
                    posfactor = 20.0
                elif abs(data.qpos[0]) > 0.3:
                    posfactor = 5.0
                elif abs(data.qpos[0]) > 0.2:
                    posfactor = 2.0

                # reward = - ((data.qpos[1] ** 2+0.01) * (data.qvel[1] ** 2+0.001)*(data.qpos[0]**2+0.01)*(data.qvel[0]**2+1)*posfactor)  # Example reward function
                reward = -cal_dis(data)
                rewards.append(reward)
                # log_probs.append(log_prob)
                done = data.time > 150  # Example condition to end episode

                ####################################
                ### commit the following line to speed up the training
                ####################################
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
                viewer.sync()

                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                ####################################
                ####################################

            # agent.update(rewards, log_probs)
            episode += 1



        # after training:
        # while viewer.is_running():
        #     step_start = time.time()
        #     action, log_prob = agent.sample_action(state)
        #     data.ctrl[0] = action
        #     mujoco.mj_step(model, data)
        #
        #     with viewer.lock():
        #         viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
        #     viewer.sync()
        #
        #     time_until_next_step = model.opt.timestep - (time.time() - step_start)
        #     if time_until_next_step > 0:
        #         time.sleep(time_until_next_step)