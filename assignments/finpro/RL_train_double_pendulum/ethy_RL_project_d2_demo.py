from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import math
import os

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import mujoco
import glfw
import cv2
import tools
import time


def init_glfw(width=800, height=600):
    if not glfw.init():
        return None
    window = glfw.create_window(width, height, "MuJoCo Simulation", None, None)
    if not window:
        glfw.terminate()
        return None
    glfw.make_context_current(window)
    return window

def cal_dis(data):
    x = data.qpos[0] + 0.6*np.sin(data.qpos[1])
    y = 0.6*np.cos(data.qpos[1])
    return (y-1.2)**2

def render(window, mujoco_model, mujoco_data):
    width, height = glfw.get_framebuffer_size(window)
    scene = mujoco.MjvScene(mujoco_model, maxgeom=1000)
    context = mujoco.MjrContext(mujoco_model, mujoco.mjtFontScale.mjFONTSCALE_150)

    mujoco.mjv_updateScene(mujoco_model, mujoco_data, mujoco.MjvOption(), None, mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(mujoco.MjrRect(0, 0, width, height), scene, context)

    buffer = np.zeros((height, width, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(buffer, None, mujoco.MjrRect(0, 0, width, height), context)
    buffer = np.flipud(buffer)
    cv2.imshow("MuJoCo Simulation", buffer)
    cv2.waitKey(1)

def calculateRev(dis):
    if dis < 0.0001:
        return 1.0
    return np.exp(100.0*dis)

if __name__ == "__main__":
    xml_path = "inverted_double_pendulum.xml"
    model_path = "temp_model_save_at_epoch_20000.pth"
    save_model_path = "none.pth"

    model, data = tools.init_mujoco(xml_path)
    window = init_glfw()

    if window:
        obs_space_dims = model.nq
        action_space_dims = model.nu
        print('nq: ', model.nq)
        print('nu: ', model.nu)
        print('state: ', data.qpos.shape[0])
        agent = tools.A3CAgent(obs_space_dims, action_space_dims, lr=3e-5, gamma=0.99)
        agent.load_model(model_path)
        # tools.load_model(agent, model_path)
        # optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)
        camera = mujoco.MjvCamera()
        camera.lookat[0] = 2.0
        camera.lookat[1] = 2.0
        camera.lookat[2] = 2.0
        # tools.load_model(agent.policy_network, model_path)
        total_num_episodes = int(500) # training epochs

        for episode in range(total_num_episodes):
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

                render(window, model, data) # commit this line to speed up the training
            # log_probs.append(log_prob)
            agent.update(rewards, log_probs, states)

        glfw.terminate()
        agent.save_model(save_model_path)
        # tools.save_model(agent, save_model_path)
