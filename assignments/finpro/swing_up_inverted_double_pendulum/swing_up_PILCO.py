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


def rollout(model, data, timesteps, pilco = None):
    X = []
    Y = []
    random = (pilco is None)
    mujoco.mj_resetData(model, data)
    data.qpos[1] = -np.pi
    state = tools.get_obs(data)
    # print("Starting State:\n", data)
    for timestep in range(timesteps):
        # if render: env.render()
        # u = np.random.uniform(-20, 20)
        u = policy(model, data, pilco, state, random)
        data.ctrl[0] = u
        mujoco.mj_step(model, data)
        new_state = tools.get_obs(data)
        X.append(np.hstack((state, u)))
        Y.append(new_state - state)
        state = new_state
    return np.stack(X), np.stack(Y)

def policy(model, data, pilco, x, random):
    if random:
        return np.random.uniform(-20, 20)
    else:
        u = pilco.compute_action(x[None, :])[0, :]
        return u.detach().cpu().numpy()

def show_rewards(rewards, folder_name):
    plt.figure()
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward with Episode')
    plt.savefig(f'{folder_name}/rewards.eps', format='eps')
    plt.show()

def generateData(model, data, timesteps = 100, depth = 10):
    X, Y = rollout(model, data, timesteps)
    for i in range(depth):
        X_, Y_ = rollout(model, data, timesteps)
        X = np.vstack((X, X_))
        Y = np.vstack((Y, Y_))
    return X, Y

import pilco
from pilco.rewards import ExponentialReward, LinearReward, CombinedRewards
from pilco.controllers import RbfController, LinearController
from pilco.models import PILCO

import tools

# 论文中的 C 矩阵
C = np.array([[1, -0.5, 0, -0.5, 0],
              [0, 0, 0.5, 0, 0.5]])

# 饱和二次项参数
W1 = 16/9 * C.T @ C
W2 = 9 * W1
t = np.zeros((1, C.shape[1]))
x_star = np.array([0, 0, 1, 0, 1])

# 双铰链项参数
a = np.array([10, 1, 1, 1, 1, 1, 1, 1, 1, 1])
b1 = np.array([-0.5, -3.5, -8*np.pi, -8*np.pi, -0.5, -3.5, -8*np.pi, -8*np.pi, -8*np.pi, -8*np.pi])
b2 = np.array([0.5, 3.5, 8*np.pi, 8*np.pi, 0.5, 3.5, 8*np.pi, 8*np.pi, 8*np.pi, 8*np.pi])


class DoubleHingeReward(nn.Module):
    def __init__(self, state_dim, a, b1, b2):
        super(DoubleHingeReward, self).__init__()
        self.state_dim = state_dim
        self.a = a
        self.b1 = b1
        self.b2 = b2

    def compute_reward(self, m, s):
        rewards = []
        for i in range(self.state_dim):
            xi = m[0, i]
            if xi < self.b1[i]:
                rewards.append(-self.a[i] * (xi - self.b1[i]))
            elif xi > self.b2[i]:
                rewards.append(self.a[i] * (xi - self.b2[i]))
            else:
                rewards.append(0)
        muR = torch.tensor(rewards).float().cuda().sum().reshape(1, 1)
        sR = torch.zeros(1, 1).cuda()
        return muR, sR


# 设置奖励函数
state_dim = 5  # 确保与 C 矩阵的维度一致
exponential_reward = ExponentialReward(state_dim, W=W1, t=t)
double_hinge_reward = DoubleHingeReward(state_dim, a=a[:state_dim], b1=b1[:state_dim], b2=b2[:state_dim])

R = CombinedRewards(state_dim, rewards=[exponential_reward, double_hinge_reward], coefs=[0.5, 0.5])

if __name__ == "__main__":
    xml_path = "inverted_swing_double_pendulum.xml"
    # current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    # current_time = f'swing_pilco_{current_time}'
    model, data = tools.init_mujoco(xml_path)
    # os.makedirs(f"{current_time}", exist_ok=True)

    obs_space_dims = 10
    action_space_dims = 1

    ######## INIT FOR PILCO ########
    controller = LinearController(state_dim=obs_space_dims, control_dim=action_space_dims)
    # R = ExponentialReward(state_dim=obs_space_dims, t=np.array([0.0, 0.0, 1.0, 0.0, 0.0]))
    m_init = np.reshape([0.0, 0.0, 0.99699654, -0.0774461, 0.0], (1, 5))
    S_init = np.diag([0.01, 0.01, 0.01, 0.01, 0.01])
    m_init = torch.from_numpy(m_init).float().cuda()
    S_init = torch.from_numpy(S_init).float().cuda()

    X,Y = generateData(model, data, timesteps = 100, depth = 10)
    pilco = PILCO(X, Y, controller=controller, horizon=40,
                  reward=R, m_init=m_init, S_init=S_init)

    ###### END INIT FOR PILCO ##########

    # agent = tools.DDPGAgent(obs_space_dims, action_space_dims, lr_a=1e-4, lr_c=1e-4, gamma=0.99, alpha=0.02,device='cuda')
    # read_model_path = "stable_2024-06-14-12-50-35/swing_up.pth"
    # save_model_path = "swing_up.pth"
    #
    # try:
    #     agent.load_model(read_model_path)
    # except FileNotFoundError:
    #     print(f"No saved model found at {read_model_path}. Starting from scratch.")
    #
    # auto_save_epochs = 20
    # total_rewards = []
    # episode = 0
    # t_limit = 25.0
    # step_count = 0

    T = 30

    for rollouts in range(20):
        pilco.optimize_models()
        pilco.optimize_policy()

        X_new, Y_new = rollout(model, data, timesteps=100, pilco=pilco)

        # multi-step prediction
        m_p = np.zeros((T, obs_space_dims))
        S_p = np.zeros((T, obs_space_dims, obs_space_dims))
        for h in range(T):
            m_h, S_h, _ = pilco.predict(m_init, S_init, h)
            m_p[h, :], S_p[h, :, :] = m_h[0, :].detach().cpu().numpy(), S_h[:, :].detach().cpu().numpy()

        for i in range(obs_space_dims):
            plt.plot(range(T - 1), m_p[0:T - 1, i], X_new[1:T, i])  # can't use Y_new because it stores differences (Dx)
            plt.fill_between(range(T - 1),
                             m_p[0:T - 1, i] - 2 * np.sqrt(S_p[0:T - 1, i, i]),
                             m_p[0:T - 1, i] + 2 * np.sqrt(S_p[0:T - 1, i, i]), alpha=0.2)
            plt.show()

        print("One iteration done")
        import pdb
        pdb.set_trace()
        # print("No of ops:", len(tf.get_default_graph().get_operations()))
        # Update dataset
        X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
        pilco.mgpr.set_XY(X, Y)
    # try:
    #     total_num_episodes = int(39)
    #     update_target_network_steps = 1000
    #
    #     while episode < total_num_episodes:
    #         rewards = []
    #         done = False
    #         data.time = 0
    #         print('The episode is:', episode)
    #
    #         # 重置环境到初始状态
    #         mujoco.mj_resetData(model, data)
    #         data.qpos[1] = -np.pi
    #         xlim = 0.75
    #         state = tools.get_obs(data)
    #
    #         while not done:
    #             step_start = time.time()
    #             action, _ = agent.sample_action(state)
    #             data.ctrl[0] = action
    #             mujoco.mj_step(model, data)
    #             next_state = tools.get_obs(data)
    #
    #             x,_,y =  data.site_xpos[0]
    #             reward = y
    #             if (reward > 1.02):
    #                 dist_penalty = 0.08 * x ** 2 + 10.0 * (y - 2) ** 2
    #                 v1, v2 = data.qvel[1:3]
    #                 # vel_penalty = 1e-3 * v1 ** 2 + 5e-3 * v2 ** 2
    #                 vel_penalty = 8e-3 * v1 ** 2 + 4e-2 * v2 ** 2
    #                 # reward = alive_bonus
    #                 reward += 2.0 - dist_penalty - vel_penalty
    #             # print(reward)
    #
    #             done = data.time > t_limit
    #             agent.store_transition(state, action, reward, next_state)
    #             state = next_state
    #             rewards.append(reward)
    #             step_count += 1
    #
    #         total_rewards.append(np.sum(np.array(rewards)))
    #         episode += 1
    #
    #         if episode % auto_save_epochs == 0:
    #             agent.save_model(f"{current_time}/temp_model_save_at_epoch_{episode}.pth")
    #
    #     agent.save_model(f"{current_time}/{save_model_path}")
    #     if total_rewards:
    #         show_rewards(total_rewards, current_time)
    #
    # except (KeyboardInterrupt, ValueError) as e:
    #     agent.save_model(f"{current_time}/autosave.pth")
    #     if total_rewards:
    #         show_rewards(total_rewards, current_time)
    #     print("Training interrupted. Model saved.")