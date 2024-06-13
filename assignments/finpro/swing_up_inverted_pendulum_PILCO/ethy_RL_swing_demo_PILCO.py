import time
import os
import torch
import numpy as np
import gymnasium as gym
import mujoco
import tools
from tools import current_reward
import matplotlib.pyplot as plt


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
    current_time = f'ddpg_{current_time}'
    model, data = tools.init_mujoco(xml_path)
    os.makedirs(f"{current_time}", exist_ok=True)

    obs_space_dims = 4  # 修改为 4 个状态变量
    action_space_dims = model.nu
    agent = tools.PILCOAgent(obs_space_dims, action_space_dims, lr=1e-3, gamma=0.98)
    agent = tools.DDPGAgent(obs_space_dims, action_space_dims, lr_a=5e-4, lr_c=5e-4, gamma=0.99, alpha=0.02)
    read_model_path = "ddpg_2024-06-14-01-40-48/temp_model_save_at_epoch_40.pth"
    save_model_path = "swing_up.pth"

    try:
        agent.load_model(read_model_path)
    except FileNotFoundError:
        print(f"No saved model found at {read_model_path}. Starting from scratch.")

    auto_save_epochs = 20
    total_rewards = []
    episode = 0
    t_limit = 12.0
    step_count = 0

    try:
        total_num_episodes = int(119)
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
                # action = action[0]
                data.ctrl[0] = action[0]
                mujoco.mj_step(model, data)
                next_state = tools.get_obs(data)

                # 使用提供的 current_reward 函数计算奖励
                reward = current_reward(next_state)

                done = data.time > t_limit
                agent.store_transition(state, action, reward, next_state)
                rewards.append(reward)
                # log_probs.append(log_prob)
                # states.append(state)
                # actions.append(action)
                # next_states.append(next_state)
                state = next_state

            total_rewards.append(np.sum(np.array(rewards)))
            for s, a, r, s_ in zip(states[:-1], actions, rewards, states[1:]):
                agent.store_transition(s, a, r, s_)
            # agent.update(rewards, log_probs, states, actions, next_states)
            episode += 1

            if episode % auto_save_epochs == 0:
                agent.save_model(f"{current_time}/temp_model_save_at_epoch_{episode}.pth")

        agent.save_model(f"{current_time}/{save_model_path}")
        if total_rewards:
            show_rewards(total_rewards, current_time)

    except (KeyboardInterrupt, ValueError) as e:
        agent.save_model(f"{current_time}/autosave.pth")
        if total_rewards:
            show_rewards(total_rewards, current_time)
        print("Training interrupted. Model saved.")
