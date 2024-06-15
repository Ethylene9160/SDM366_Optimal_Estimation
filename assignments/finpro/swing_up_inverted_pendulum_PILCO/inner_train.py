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
    current_time = f'inner_{current_time}'

    model, data = tools.init_mujoco(xml_path)
    os.makedirs(f"{current_time}", exist_ok=True)

    obs_space_dims = 4
    action_space_dims = model.nu
    agent = tools.PILCOAgent(obs_space_dims, action_space_dims, gamma=0.98)
    read_model_path = "2024-06-09-19-52-19/swing_up.pth"
    save_model_path = "swing_up.pth"

    try:
        agent.load_model(read_model_path)
    except FileNotFoundError:
        print(f"No saved model found at {read_model_path}. Starting from scratch.")

    auto_save_epochs = 500
    total_rewards = []
    episode = 0
    t_limit = 1.0
    step_count = 0

    # 在主函数中单独训练 gp_model
    states = []
    actions = []
    next_states = []
    total_num_episodes = 1000
    try:
        while episode < total_num_episodes:
            rewards = []
            log_probs = []
            done = False
            data.time = 0
            print('The episode is:', episode)

            # 重置环境到初始状态
            mujoco.mj_resetData(model, data)
            data.qpos[1] = -np.pi
            tools.chaotic_state(data)
            xlim = 0.75
            state = tools.get_obs(data)

            while not done:
                step_start = time.time()
                action, log_prob = agent.sample_action(state)
                data.ctrl[0] = action
                mujoco.mj_step(model, data)
                next_state = tools.get_obs(data)
                reward = -0.1

                if abs(next_state[0]) > xlim or data.time >= t_limit:
                    reward = -5.0
                elif abs(next_state[1]) < 0.2 and abs(next_state[2]) < 0.8 and abs(next_state[3]) < 0.8:
                    reward = 10.0 - 0.01 * data.qpos[0] ** 2 - 0.01 * data.qvel[0] ** 2

                done = data.time > t_limit

                rewards.append(reward)
                log_probs.append(log_prob)
                states.append(state.copy())
                actions.append(action)
                next_states.append(next_state.copy())

                state = next_state

            total_rewards.append(np.sum(np.array(rewards)))
            episode += 1

            if episode % auto_save_epochs == 0:
                agent.save_model(f"{current_time}/temp_model_save_at_epoch_{episode}.pth")

        agent.train_gp_model_only(states, actions, next_states)
        agent.save_model(f"{current_time}/{save_model_path}")
        if total_rewards:
            show_rewards(total_rewards, current_time)
    except KeyboardInterrupt:
        agent.train_gp_model_only(states, actions, next_states)
        agent.save_model(f"{current_time}/{save_model_path}")
        if total_rewards:
            show_rewards(total_rewards, current_time)
