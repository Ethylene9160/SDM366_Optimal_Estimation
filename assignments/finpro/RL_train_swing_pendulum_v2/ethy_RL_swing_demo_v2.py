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

import tools

if __name__ == "__main__":
    xml_path = "inverted_swing_pendulum.xml"
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    model, data = tools.init_mujoco(xml_path)
    os.makedirs(f"{current_time}", exist_ok=True)

    obs_space_dims = 4
    action_space_dims = model.nu
    agent = tools.A2CAgent(obs_space_dims, action_space_dims, lr=5e-4, gamma = 0.95)
    model_path = ""
    read_model_path = "2024-06-07-22-27-53/a2c_policy_v2.pth"
    save_model_path = "a2c_policy_v2.pth"
    try:
        agent.load_model(read_model_path)
    except FileNotFoundError:
        print(f"No saved model found at {read_model_path}. Starting from scratch.")

    auto_save_epochs = 200

    try:
        # create viewer
        with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
            total_num_episodes = int(5000)
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
                data.qpos[1] = -np.pi
                alive_bonus = 100.0
                while not done:
                    step_start = time.time()
                    state = tools.get_obs(data)
                    action, log_prob = agent.sample_action(state)
                    data.ctrl[0] = action
                    mujoco.mj_step(model, data)
                    ######  Calculate the Reward #######
                    # reward = -0.4*data.qpos[0]**2 - 0.6*data.qpos[1]**2 - 0.001*data.qvel[1]**2
                    reward = -0.1*(5*data.qpos[1]**2+data.qpos[0]**2+0.05*action**2)
                    if abs(data.qpos[0]) > 1.8:
                        reward -= alive_bonus
                        # reward += 0.001*action**2
                    # if abs(data.qpos[1]) < 1.5:
                    #     reward += 1.0 - 0.005*action**2  - 0.1*data.qvel[1]**2 - 0.1*data.qvel[0]**2 - 0.1*data.qpos[0]**2
                        # reward -=
                    ###### End. The same as the official model ########

                    rewards.append(reward)
                    log_probs.append(log_prob)

                    states.append(state.copy())
                    done = data.time > 25  # Example condition to end episode

                    ####################################
                    ### commit the following line to speed up the training
                    ####################################
                    # with viewer.lock():
                    #     viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
                    # viewer.sync()
                    #
                    # time_until_next_step = model.opt.timestep - (time.time() - step_start)
                    # if time_until_next_step > 0:
                    #     time.sleep(time_until_next_step)
                    ####################################
                    ####################################

                agent.update(rewards, log_probs, states)
                episode += 1
                if episode % auto_save_epochs == 0:
                    agent.save_model(f"{current_time}/temp_model_save_at_epoch_{episode}.pth")
            agent.save_model(f"{current_time}/{save_model_path}")
    except KeyboardInterrupt:
        agent.save_model(f"{current_time}/autosave.pth")
        print("Training interrupted. Model saved.")
