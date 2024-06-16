import time
import os
import torch
import numpy as np
import gymnasium as gym
import mujoco
import tools

import matplotlib.pyplot as plt


def show_rewards(rewards, folder_name):
    plt.figure()
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward with Episode')
    plt.savefig(f'{folder_name}/rewards.eps', format='eps')
    plt.show()

def current_reward(state, xlim=0.85):
    '''
    state[0]: position of cart
    state[1]: theta
    state[2]: v of the cart
    state[3]: omega
    '''
    reward =-1.8 * state[1] ** 2 -  0.002 * state[3] ** 2 - 0.2* state[0] ** 2
    if abs(state[0]) > xlim:
        reward = -20.0
    elif abs(state[1]) < 0.20:
        reward += 3.5
    return reward

train_mode = False # 调整是否进行训练。

if __name__ == "__main__":
    xml_path = "inverted_swing_pendulum.xml"
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    current_time = f'ddpg_{current_time}'
    model, data = tools.init_mujoco(xml_path)
    if train_mode:
        os.makedirs(f"{current_time}", exist_ok=True)

    obs_space_dims = 4  # 修改为 4 个状态变量
    action_space_dims = model.nu
    # agent = tools.PILCOAgent(obs_space_dims, action_space_dims, lr=1e-3, gamma=0.98)
    agent = tools.DDPGAgent(obs_space_dims, action_space_dims, lr_a=8e-5, lr_c=8e-5, gamma=0.99, alpha=0.02)
    read_model_path = "ddpg_2024-06-14-15-34-28/swing_up.pth"
    save_model_path = "swing_up.pth"

    try:
        agent.load_model(read_model_path)
    except FileNotFoundError:
        print(f"No saved model found at {read_model_path}. Starting from scratch.")

    auto_save_epochs = 120
    total_rewards = []
    episode = 0
    t_limit = 12.0
    step_count = 0

    try:
        total_num_episodes = int(899)
        update_target_network_steps = 1000

        while episode < total_num_episodes:
            rewards = []
            log_probs = []
            states = []
            actions = []
            next_states = []
            done = False
            data.time = 0
            print('The episode is:', episode)

            mujoco.mj_resetData(model, data)
            data.qpos[1] = -np.pi
            # xlim = 0.75
            state = tools.get_obs(data)
            while not done:
                step_start = time.time()
                action, _ = agent.sample_action(state)
                data.ctrl[0] = action[0]
                mujoco.mj_step(model, data)
                next_state = tools.get_obs(data)

                reward = current_reward(next_state)

                done = data.time > t_limit
                agent.store_transition(state, action, reward, next_state)
                rewards.append(reward)
                state = next_state

            total_rewards.append(np.sum(np.array(rewards)))
            if train_mode:
                for s, a, r, s_ in zip(states[:-1], actions, rewards, states[1:]):
                    agent.store_transition(s, a, r, s_)
            # agent.update(rewards, log_probs, states, actions, next_states)
            episode += 1

            if episode % auto_save_epochs == 0 and train_mode:
                agent.save_model(f"{current_time}/temp_model_save_at_epoch_{episode}.pth")
        if train_mode:
            agent.save_model(f"{current_time}/{save_model_path}")
        if total_rewards:
            show_rewards(total_rewards, current_time)

    except (KeyboardInterrupt, ValueError) as e:
        if train_mode:
            agent.save_model(f"{current_time}/autosave.pth")
        if total_rewards:
            show_rewards(total_rewards, current_time)
        print("Training interrupted. Model saved.")
