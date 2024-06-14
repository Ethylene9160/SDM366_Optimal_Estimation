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
    xml_path = "inverted_swing_double_pendulum.xml"
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    current_time = f'swing_{current_time}'
    model, data = tools.init_mujoco(xml_path)
    os.makedirs(f"{current_time}", exist_ok=True)

    obs_space_dims = 6
    action_space_dims = model.nu
    agent = tools.DDPGAgent(obs_space_dims, action_space_dims, lr_a=1e-4, lr_c=1e-4, gamma=0.99, alpha=0.02,device='cuda')
    read_model_path = "2024-06-14-11-00-24/swing_up.pth"
    save_model_path = "swing_up.pth"

    try:
        agent.load_model(read_model_path)
    except FileNotFoundError:
        print(f"No saved model found at {read_model_path}. Starting from scratch.")

    auto_save_epochs = 20
    total_rewards = []
    episode = 0
    t_limit = 25.0
    step_count = 0

    try:
        total_num_episodes = int(39)
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
                action, _ = agent.sample_action(state)
                data.ctrl[0] = action
                mujoco.mj_step(model, data)
                next_state = tools.get_obs(data)

                x,_,y =  data.site_xpos[0]
                reward = y
                if (reward > 1.02):
                    dist_penalty = 0.08 * x ** 2 + 10.0 * (y - 2) ** 2
                    v1, v2 = data.qvel[1:3]
                    # vel_penalty = 1e-3 * v1 ** 2 + 5e-3 * v2 ** 2
                    vel_penalty = 8e-3 * v1 ** 2 + 4e-2 * v2 ** 2
                    # reward = alive_bonus
                    reward += 2.0 - dist_penalty - vel_penalty
                # print(reward)

                done = data.time > t_limit
                agent.store_transition(state, action, reward, next_state)
                state = next_state
                rewards.append(reward)
                step_count += 1

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