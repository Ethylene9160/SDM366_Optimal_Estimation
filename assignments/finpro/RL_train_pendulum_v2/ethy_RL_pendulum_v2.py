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



if __name__ == "__main__":
    xml_path = "inverted_pendulum.xml"
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    model, data = tools.init_mujoco(xml_path)
    os.makedirs(f"{current_time}", exist_ok=True)

    obs_space_dims = 4
    action_space_dims = model.nu
    agent = tools.A2CAgent(obs_space_dims, action_space_dims, lr=3e-4, gamma = 0.99)
    # model_path = ""
    read_model_path = "models/ethy_official_model_fimal.pth"
    save_model_path = "a2c_policy_v2.pth"
    try:
        agent.load_model(read_model_path)
    except FileNotFoundError:
        print(f"No saved model found at {read_model_path}. Starting from scratch.")

    auto_save_epochs = 1000

    try:

        # create viewer
        with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
            total_num_episodes = int(20000)
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

                    ######  Calculate the Reward #######
                    reward = 1.0
                    ###### End. The same as the official model ########
                    ###### SUPPLIMENT: to better the performance ########
                    pennity = 0.8*data.qpos[0]**2+0.001*data.qvel[0]**2+0.1*data.qvel[1]**2
                    reward -= pennity
                    ###### End calculating the reward #######

                    rewards.append(reward)
                    log_probs.append(log_prob)
                    states.append(state.copy())
                    done = data.time > 20 or abs(data.qpos[1]) > 0.18  # Example condition to end episode

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
