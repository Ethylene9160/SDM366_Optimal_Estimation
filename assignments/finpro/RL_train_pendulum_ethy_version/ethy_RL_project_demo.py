from __future__ import annotations

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


def init_glfw(width=640, height=480):
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
    return x**2+(y-0.6)**2

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
    xml_path = "inverted_pendulum.xml"
    model_path = "new_policy1.pth"
    save_model_path = "new_policy1.pth"

    model, data = tools.init_mujoco(xml_path)
    window = init_glfw()

    if window:
        obs_space_dims = model.nq
        action_space_dims = model.nu
        agent = tools.Agent(obs_space_dims, action_space_dims, lr=8e-4, gama=0.85)
        tools.load_model(agent.policy_network, model_path)
        total_num_episodes = int(2000) # training epochs

        for episode in range(total_num_episodes):
            rewards = []
            log_probs = []
            done = False
            data.time = 0

            # reset the state
            mujoco.mj_resetData(model, data)
            print('init angle: ', data.qpos[1])
            print('episode:', episode)
            # counter = 0
            action = None
            log_prob = None
            while not done:
                step_start = time.time()
                state = data.qpos
                action, log_prob = agent.sample_action(state)
                data.ctrl[0] = action
                mujoco.mj_step(model, data)
                reward = -calculateRev(cal_dis(data))
                # reward = -  (data.qpos[1] ** 2 * data.qvel[1] ** 2)*posfactor  # Example reward function
                rewards.append(reward)
                log_probs.append(log_prob)
                # print(cal_dis(data))
                done = data.time > 1.0  # Example condition to end episode
                render(window, model, data) # commit this line to speed up the training
            log_probs.append(log_prob)
            agent.update(rewards, log_probs)

        glfw.terminate()
        tools.save_model(agent.policy_network, save_model_path)
