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
import sys


import tools

train_mode = False # Set to True to run the trained model

if __name__ == "__main__":
    xml_path = "inverted_double_pendulum.xml"
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    model, data = tools.init_mujoco(xml_path)
    if train_mode:
        os.makedirs(f"{current_time}", exist_ok=True)

    obs_space_dims = 10
    action_space_dims = model.nu
    agent = tools.A2CAgent(obs_space_dims, action_space_dims, lr=6e-5, gamma = 0.99, device='cuda')

    read_model_path = "models/ethy_official_model4.pth"
    save_model_path = "a2c_policy_v2.pth"
    try:
        agent.load_model(read_model_path)
    except FileNotFoundError:
        print(f"No saved model found at {read_model_path}. Starting from scratch.")

    auto_save_epochs = 200

    try:

        # create viewer
        with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
            total_num_episodes = int(2000)
            episode = 0
            while viewer.is_running() and episode < total_num_episodes:
                rewards = []
                log_probs = []
                states = []
                done = False
                data.time = 0
                print('The episode is:', episode)

                # 重置环境到初始状态
                mujoco.mj_resetData(model, data)
                tools.random_state(data)
                alive_bonus = 10.0
                while not done:
                    step_start = time.time()
                    state = tools.get_obs(data)
                    action, log_prob = agent.sample_action(state)
                    data.ctrl[0] = action
                    mujoco.mj_step(model, data)

                    x, _, y = data.site_xpos[0]
                    if train_mode:
                        ######  Calculate the Reward #######
                        # dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
                        dist_penalty = 0.08*x**2+10.0*(y-2)**2
                        v1, v2 = data.qvel[1:3]
                        # vel_penalty = 1e-3 * v1 ** 2 + 5e-3 * v2 ** 2
                        vel_penalty = 8e-3 * v1 ** 2 + 4e-2 * v2 ** 2
                        reward = alive_bonus - dist_penalty - vel_penalty
                        ###### End. The same as the official model ########

                        rewards.append(reward)
                        log_probs.append(log_prob)
                        states.append(state.copy())
                    done = data.time > 25 or y < 0.95  # Example condition to end episode


                    ####################################
                    ### commit the following line to speed up the training
                    ####################################
                    if not train_mode:
                        with viewer.lock():
                            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
                        viewer.sync()

                        time_until_next_step = model.opt.timestep - (time.time() - step_start)
                        if time_until_next_step > 0:
                            time.sleep(time_until_next_step)
                    ####################################
                    ####################################
                if train_mode:
                    agent.update(rewards, log_probs, states)
                episode += 1
                if episode % auto_save_epochs == 0 and train_mode:
                    agent.save_model(f"{current_time}/temp_model_save_at_epoch_{episode}.pth")
            if train_mode:
                agent.save_model(f"{current_time}/{save_model_path}")
    except KeyboardInterrupt:
        if train_mode:
            agent.save_model(f"{current_time}/autosave.pth")
        print("Training interrupted. Model saved.")
