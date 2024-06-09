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


def is_stable(data):
    if abs(data.qpos[1]) < 0.35 and abs(data.qvel[0]) < 5.5 and abs(data.qvel[1]) <3.99:
        return True
    return False

def is_unstable(data):
    if abs(data.qpos[1]) > 0.8:
        return True
    return False

def get_action(agent, action_space, stable_state):
    if stable_state:
        action, _ = agent.sample_action(state)
    else:
        action = agent.sample_action(state)
        action = action_space[action]
    return action


import tools
if __name__ == "__main__":
    xml_path = "inverted_swing_pendulum.xml"
    # stable_model_path = "stable_2024-06-09-20-49-46/autosave.pth"
    # swing_model_path = "v62_2024-06-09-20-22-21/temp_model_save_at_epoch_60.pth"
    stable_model_path = 'models/ethy_official_stable_model.ethy'
    swing_model_path = 'models/ethy_official_swing_model.ethy'
    model, data = tools.init_mujoco(xml_path)

    stable_state = False
    obs_space_dims = 6
    action_space_dims = model.nu
    print('nq: ', model.nq)
    print('nu: ', model.nu)
    print('state: ', data.qpos.shape[0])
    action_space = [-15.0, -5.0, -1.2, -0.4, 0, 0.4, 1.2, 5.0, 15.0]
    swing_agent = tools.DQNAgent(obs_space_dims, action_space, gamma=0.98)
    swing_agent.load_model(swing_model_path)
    stable_agent = tools.A2CAgent(obs_space_dims, action_space_dims, lr=0.000, gamma=0.99)
    stable_agent.load_model(stable_model_path)
    # agent = tools.A2CAgent(obs_space_dims, action_space_dims, lr=0.000, gamma=0.99)
    # agent.load_model(model_path)
    agent = swing_agent
    total_num_episodes = int(10)  # training epochs
    time_records = []
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
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
            while not done:
                step_start = time.time()

                action = get_action(agent, action_space, stable_state)
                data.ctrl[0] = action
                mujoco.mj_step(model, data)
                state = tools.get_obs(data)

                if stable_state == False:
                    if is_stable(data):
                        stable_state = True
                        agent = stable_agent
                        print('stable')
                else:
                    if is_unstable(data):
                        stable_state = False
                        agent = swing_agent
                        print('swing')

                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
                viewer.sync()

                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                i += 1
                # if i % 100 == 0:
                #     print(f'state: {state}, action: {action}, next_state: {next_state}')
                done = data.time > 25 and (not stable_state) # Example condition to end episode
                # state = next_state
            episode += 1
