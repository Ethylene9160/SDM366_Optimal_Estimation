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

    obs_space_dims = 5
    action_space_dims = model.nu
    agent = tools.A2CAgent(obs_space_dims, action_space_dims, lr=6e-6, gamma = 0.99)
    model_path = ""
    read_model_path = "2024-06-08-03-37-35/temp_model_save_at_epoch_55000.pth"
    save_model_path = "new_a2c_policy_v2.pth"
    try:
        agent.load_model(read_model_path)
    except FileNotFoundError:
        print(f"No saved model found at {read_model_path}. Starting from scratch.")

    auto_save_epochs = 200

    total_rewards = []
    try:
        # create viewer
        with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
            total_num_episodes = int(1000)
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
                # data.qpos[1] = -np.pi
                alive_bonus = 100.0
                xlim = 2.45
                while not done:
                    step_start = time.time()
                    state = tools.get_obs(data)
                    action, log_prob = agent.sample_action(state)
                    data.ctrl[0] = action
                    mujoco.mj_step(model, data)
                    ######  Calculate the Reward #######
                    rt = (1-state[4])/2
                    rx = math.cos(data.qpos[0]*np.pi/xlim/2)

                    reward = rt*rx+0.1
                    if state[4] < 0.95:
                        reward += 10 - 0.5*data.qpos[0]**2 - 0.05*data.qvel[0]**2
                    elif state[4] < -0.85:
                        reward += 5
                    elif state[4] < -0.37:
                        reward += 1
                    # reward = -state[4]
                    # reward = math.exp(reward)
                    #
                    # if state[4] <= -0.85:
                    #     reward = reward + 10 - 5*data.xpos[0]**2-0.5*data.qvel[1]**2-0.1*data.qvel[0]**2
                    # elif state[4] < 0:
                    #     reward *= 10
                    # elif state[4] < 0.5:
                    #     reward *= 5
                    # if abs(data.qpos[0]) > 0.9 and state[4] > -0.85:
                    #     reward = -1.0

                    ###### End. The same as the official model ########

                    rewards.append(reward)
                    log_probs.append(log_prob)

                    states.append(state.copy())
                    done = data.time > 20 or abs(data.qpos[0]) > xlim # Example condition to end episode

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
                total_rewards.append(np.sum(np.array(rewards)))
                episode += 1
                if episode % auto_save_epochs == 0:
                    agent.save_model(f"{current_time}/temp_model_save_at_epoch_{episode}.pth")
            agent.save_model(f"{current_time}/{save_model_path}")
            show_rewards(total_rewards, current_time)
    except KeyboardInterrupt or ValueError:
        agent.save_model(f"{current_time}/autosave.pth")
        show_rewards(total_rewards, current_time)
        print("Training interrupted. Model saved.")
