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

    model, data = tools.init_mujoco(xml_path)

    obs_space_dims = 6
    action_space_dims = model.nu
    agent = tools.DDPGAgent(obs_space_dims, 1, a_bound=2)
    read_model_path = "stable_2024-06-14-14-09-04/swing_up.pth"
    # save_model_path = "swing_up.pth"
    try:
        agent.load_model(read_model_path)
    except FileNotFoundError:
        print(f"No saved model found at {read_model_path}. Starting from scratch.")

    total_rewards = []

    # create viewer
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        total_num_episodes = int(1000)
        episode = 0
        while viewer.is_running() and episode < total_num_episodes:
            rewards = []
            states = []
            actions = []
            next_states = []
            dones = []
            done = False
            data.time = 0
            print('The episode is:', episode)

            # 重置环境到初始状态
            mujoco.mj_resetData(model, data)
            # data.qpos[1] = -np.pi
            tools.random_state(data)
            alive_bonus = 100.0
            xlim = 1.5
            i = 0
            state = tools.get_obs(data)
            print('init qpos:', data.qpos)
            while not done:
                step_start = time.time()
                action, _ = agent.sample_action(state)
                data.ctrl[0] = action
                mujoco.mj_step(model, data)
                next_state = tools.get_obs(data)

                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
                viewer.sync()

                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                i += 1
                if i % 100 == 0:
                    x, _, y = data.site_xpos[0]
                    print(f'state: {state}, action: {action}, next_state: {next_state}, y: {y}')
                done = data.time > 25 # Example condition to end episode
                state = next_state
            episode += 1
