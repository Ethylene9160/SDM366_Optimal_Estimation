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



def init_mujoco(model_path):
    mujoco_model = mujoco.MjModel.from_xml_path(model_path)
    mujoco_data = mujoco.MjData(mujoco_model)
    return mujoco_model, mujoco_data


def cal_dis(data):
    x = data.qpos[0] + 0.6*np.sin(data.qpos[1])
    y = 0.6*np.cos(data.qpos[1])
    return (y-1.6)**2

def _cal_top(data):
    theta1 = data.qpos[0]
    theta2 = data.qpos[1]+theta1
    x = 0.6*np.sin(theta1)+0.6*np.sin(theta2)
    y = 0.6*np.cos(theta1)+0.6*np.cos(theta2)
    return x,y

if __name__ == "__main__":
    xml_path = "inverted_double_pendulum.xml"

    model, data = init_mujoco(xml_path)

    obs_space_dims = model.nq
    action_space_dims = model.nu
    agent = tools.A3CAgent(obs_space_dims, action_space_dims,lr=1e-5, gamma = 0.99)
    # model_path = ""
    save_model_path = "a3c_policy1.pth"
    # try:
    #     agent.load_model(save_model_path)
    # except FileNotFoundError:
    #     print(f"No saved model found at {save_model_path}. Starting from scratch.")

    # create viewer
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        total_num_episodes = int(5e4)
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

            while not done:
                step_start = time.time()
                state = data.qpos
                action, log_prob = agent.sample_action(state)
                data.ctrl[0] = action
                mujoco.mj_step(model, data)

                ######  Calculate the Reward #######
                x, _, y = data.site_xpos[0]
                dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
                v1, v2 = data.qvel[1:3]
                vel_penalty = 1e-3 * v1 ** 2 + 5e-3 * v2 ** 2
                alive_bonus = 10.0
                reward = alive_bonus - dist_penalty - vel_penalty
                ###### End. The same as the official model ########

                rewards.append(reward)
                log_probs.append(log_prob)
                states.append(state.copy())
                done = data.time > 12 or y < 1.0  # Example condition to end episode

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
            episode += 1
            if episode % 10000 == 0:
                agent.save_model(f"temp_model_save_at_epoch_{episode}.pth")
        agent.save_model(save_model_path)
