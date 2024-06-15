from __future__ import annotations

import math
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
import matplotlib.pyplot as plt

def show_rewards(rewards, folder_name):
    plt.figure()
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward with Episode')
    plt.savefig(f'{folder_name}/rewards.eps', format='eps')
    plt.show()

import tools
if __name__ == "__main__":
    xml_path = "inverted_swing_pendulum.xml"
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    model, data = tools.init_mujoco(xml_path)
    os.makedirs(f"{current_time}", exist_ok=True)

    obs_space_dims = 6
    action_space_dims = model.nu
    action_space = [-15.0, -5.0, -1.2, -0.4, 0, 0.4, 1.2, 5.0, 15.0]
    agent = tools.DQNAgent(obs_space_dims, action_space, gamma=0.98)
    read_model_path = "2024-06-09-19-52-19/swing_up.pth"
    save_model_path = "swing_up.pth"

    try:
        agent.load_model(read_model_path)
    except FileNotFoundError:
        print(f"No saved model found at {read_model_path}. Starting from scratch.")

    auto_save_epochs = 20
    total_rewards = []
    episode = 0
    t_limit = 10.0
    step_count = 0

    try:
        total_num_episodes = int(139)
        update_target_network_steps = 1000

        while episode < total_num_episodes:
            rewards = []
            done = False
            data.time = 0
            print('The episode is:', episode)

            # 重置环境到初始状态
            mujoco.mj_resetData(model, data)
            data.qpos[1] = -np.pi
            xlim = 0.75
            state = tools.get_obs(data)

            while not done:
                step_start = time.time()
                action_idx = agent.sample_action(state)
                action = action_space[action_idx]
                data.ctrl[0] = action
                mujoco.mj_step(model, data)
                next_state = tools.get_obs(data)
                reward = -0.1

                if abs(next_state[0]) > xlim or data.time >= t_limit:
                    reward = -5.0
                elif abs(next_state[1]) < 0.2 and abs(next_state[2]) < 0.8 and abs(next_state[3]) < 0.8:
                    reward = 10.0 - 0.01 * data.qpos[0] ** 2 - 0.01 * data.qvel[0] ** 2

                done = data.time > t_limit
                agent.remember(state, action_idx, reward, next_state, done)
                agent.replay()
                state = next_state
                rewards.append(reward)

                step_count += 1
                if step_count == update_target_network_steps:
                    agent.update_target_network()
                    step_count = 0

            total_rewards.append(np.sum(np.array(rewards)))
            episode += 1

            if episode % auto_save_epochs == 0:
                agent.save_model(f"{current_time}/temp_model_save_at_epoch_{episode}.pth")

        agent.save_model(f"{current_time}/{save_model_path}")
        if total_rewards:
            show_rewards(total_rewards, current_time)

    except (KeyboardInterrupt, ValueError) as e:
        agent.save_model(f"{current_time}/autosave.pth")
        if total_rewards:
            show_rewards(total_rewards, current_time)
        print("Training interrupted. Model saved.")