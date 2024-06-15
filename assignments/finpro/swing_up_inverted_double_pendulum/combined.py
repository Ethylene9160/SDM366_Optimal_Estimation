
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
import tools

def cal_action(is_stable, agent, data):
    if is_stable:
        u,_ = agent.sample_action(tools.get10obs(data))
    else:
        u,_ = agent.sample_action(tools.get_obs(data))
    # action, _ = agent.sample_action(state)
    return u

def is_stable(data):
    if abs(data.qpos[1]) < 0.2 and\
        abs(data.qpos[2]) < 0.2 and\
        abs(data.qvel[0]) < 0.5 and\
        abs(data.qvel[1]) < 1.0 and\
        abs(data.qvel[2]) < 1.0:
        return True
    return False
def is_unstable(data):
    _,_,z = data.site_xpos[0]
    return z < 0.8

if __name__ == "__main__":
    xml_path = "inverted_swing_double_pendulum.xml"
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    model, data = tools.init_mujoco(xml_path)

    obs_space_dims = 6
    action_space_dims = model.nu

    swing_agent = tools.DDPGAgent(obs_space_dims, 1,device='cuda')
    stable_agent = tools.A2CAgent(10, 1)
    swing_path = "swing_2024-06-15-01-41-03/temp_model_save_at_epoch_100.pth"
    stable_path = "models/ethy_official_double_stable.pth"
    # save_model_path = "swing_up.pth"
    try:
        stable_agent.load_model(stable_path)
        swing_agent.load_model(swing_path)
    except FileNotFoundError:
        print(f"No saved model found at {swing_path} or {stable_path}. Starting from scratch.")

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
            data.qpos[1] = -np.pi
            alive_bonus = 100.0
            xlim = 1.5
            i = 0
            state = tools.get_obs(data)
            print('init qpos:', data.qpos)

            isStable = False
            agent = swing_agent

            while not done:
                step_start = time.time()
                action, _ = agent.sample_action(state)
                data.ctrl[0] = action
                mujoco.mj_step(model, data)
                next_state = tools.get_obs(data)

                if isStable:
                    if is_unstable(data):
                        agent = swing_agent
                        isStable = False
                        print('unstable')
                else:
                    if is_stable(data):
                        agent = stable_agent
                        isStable = True
                        print('stable')

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