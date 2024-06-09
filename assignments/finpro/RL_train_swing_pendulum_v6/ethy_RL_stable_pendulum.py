from __future__ import annotations

import os

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.distributions.normal import Normal
import mujoco
import mujoco.viewer
# import glfw
import time

import tools

def show_rewards(rewards, folder_name):
    plt.figure()
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward with Episode')
    plt.savefig(f'{folder_name}/rewards.eps', format='eps')
    plt.show()


if __name__ == "__main__":
    xml_path = "inverted_swing_pendulum.xml"
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    model, data = tools.init_mujoco(xml_path)
    os.makedirs(f"stable_{current_time}", exist_ok=True)

    obs_space_dims = 6
    action_space_dims = model.nu
    agent = tools.A2CAgent(obs_space_dims, action_space_dims, lr=8e-4, gamma = 0.99)
    # model_path = ""
    read_model_path = "stable_2024-06-09-21-03-17/autosave.pth"
    save_model_path = "a2c_policy_v2.pth"
    try:
        agent.load_model(read_model_path)
    except FileNotFoundError:
        print(f"No saved model found at {read_model_path}. Starting from scratch.")

    auto_save_epochs = 1000
    total_reward = []
    try:

        # create viewer
        with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
            total_num_episodes = int(19999)
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
                state = tools.get_obs(data)
                while not done:
                    step_start = time.time()
                    action, log_prob = agent.sample_action(state)
                    data.ctrl[0] = action
                    mujoco.mj_step(model, data)
                    state = tools.get_obs(data)
                    ######  Calculate the Reward #######
                    reward = 1.0
                    ###### End. The same as the official model ########
                    pennity = 0.82*data.qpos[0]**2+0.001*data.qvel[0]**2+0.1*data.qvel[1]**2
                    reward -= pennity

                    rewards.append(reward)
                    log_probs.append(log_prob)
                    states.append(state.copy())
                    done = data.time > 20 or abs(data.qpos[1]) > 0.32  # Example condition to end episode

                    # 获取tip的世界坐标
                    # tip_xpos = data.site_xpos[model.site('tip').id]
                    # print("Tip Position:", tip_xpos)
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
                total_reward.append(np.sum(np.array(rewards)))
                episode += 1
                if episode % auto_save_epochs == 0:
                    agent.save_model(f"stable_{current_time}/temp_model_save_at_epoch_{episode}.pth")
            agent.save_model(f"stable_{current_time}/{save_model_path}")
            if total_reward:
                show_rewards(total_reward, f'stable_{current_time}')
    except KeyboardInterrupt:
        agent.save_model(f"stable_{current_time}/autosave.pth")
        print("Training interrupted. Model saved.")
        if total_reward:
            show_rewards(total_reward, f'stable_{current_time}')
