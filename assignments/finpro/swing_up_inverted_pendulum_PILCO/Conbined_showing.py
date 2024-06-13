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
    '''
    If the state of the pendulum is:
    abs(x) < 0.55
    abs(theta) < 0.24
    abs(v) < 1.8
    abs(omega) < 1.8
    Then it will be stable.
    '''
    if abs(data.qpos[1]) < 0.4 and abs(data.qvel[0]) < 2.0 and abs(data.qvel[1]) <2.0:
        return True
    return False

def is_unstable(data):
    if abs(data.qpos[1]) > 0.85:
        return True
    return False

def get_action(agent, data, stable_state):
    if stable_state:
        obs = tools.get6obs(data)
    else:
        obs = tools.get_obs(data)
    return agent.sample_action(obs)[0]


import tools
if __name__ == "__main__":
    xml_path = "inverted_swing_pendulum.xml"
    stable_model_path = 'models/ethy_official_stable.ethy'
    swing_model_path = 'models/ethy_official_swing_up.ethy'
    model, data = tools.init_mujoco(xml_path)

    obs_space_dims = 6
    action_space_dims = model.nu
    print('nq: ', model.nq)
    print('nu: ', model.nu)
    print('state: ', data.qpos.shape[0])
    action_space = [-15.0, -5.0, -1.2, -0.4, 0, 0.4, 1.2, 5.0, 15.0]
    swing_agent = tools.DDPGAgent(4, 1)
    swing_agent.load_model(swing_model_path)
    stable_agent = tools.A2CAgent(obs_space_dims, action_space_dims, lr=0.000, gamma=0.99)
    stable_agent.load_model(stable_model_path)
    # agent = tools.A2CAgent(obs_space_dims, action_space_dims, lr=0.000, gamma=0.99)
    # agent.load_model(model_path)
    agent = swing_agent
    total_num_episodes = int(10)  # training epochs
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

            agent = swing_agent
            stable_state = False

            # 重置环境到初始状态
            mujoco.mj_resetData(model, data)
            data.qpos[1] = -np.pi
            alive_bonus = 100.0
            xlim = 1.5
            i = 0
            state = tools.get_obs(data)
            while not done:
                step_start = time.time()

                action = get_action(agent, data, stable_state)
                # print('action:', action)
                data.ctrl[0] = action
                mujoco.mj_step(model, data)
                state = tools.get6obs(data)

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
                done = data.time > 6 #and (not stable_state) # Example condition to end episode
                # state = next_state
            episode += 1
