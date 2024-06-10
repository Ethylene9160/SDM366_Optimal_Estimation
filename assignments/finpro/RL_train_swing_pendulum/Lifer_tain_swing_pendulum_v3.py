from __future__ import annotations

import os
import time
import random

import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

import tools


# Environment reward function
def calculate_reward_a(state_list):
    """Calculate the reward based on the given state and done flag."""
    x, theta, sin_theta, cos_theta, x_dot, theta_dot = state_list

    # calculate theta in [-pi, pi] by sin and cos
    theta = np.arctan2(sin_theta, cos_theta)

    # r = 0
    # if abs(theta) >= 0.15 or abs(x) >= 0.5:
    #     r = -1.0

    p = 0.8 * theta ** 2 + 0.001 * theta_dot ** 2 + 0.1 * x_dot ** 2
    r = 1.0 - p

    return r


# Environment reward function
def calculate_reward_b(state_list):
    """Calculate the reward based on the given state and done flag."""
    x, theta, sin_theta, cos_theta, x_dot, theta_dot = state_list

    # calculate theta in [-pi, pi] by sin and cos
    theta = np.arctan2(sin_theta, cos_theta)

    if abs(theta) >= 0.75:
        r = -(0.2 * x ** 2 + 0.6 * theta ** 2 - 0.001 * theta_dot ** 2)
    else:
        r = -(0.1 * x ** 2 + 0.6 * theta ** 2 + 0.001 * theta_dot ** 2)

    if abs(x) < 0.75 and abs(theta) < 0.3 and abs(x_dot) < 0.8 and abs(theta_dot) < 0.8:
        r += 10
        if abs(x) < 0.55:
            r += 2 * (2 - abs(theta)) ** 2

    if abs(x) > 0.75:
        r -= 10

    if abs(x) > 1.25:
        r -= 40

    return r


if __name__ == "__main__":
    xml_path = "Lifer_inverted_swing_pendulum.xml"
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    model, data = tools.init_mujoco(xml_path)

    obs_space_dims = 6
    action_space_dims = model.nu

    agent_b = tools.DQNAgent(obs_space_dims, action_space_dims, lr=3e-4, gamma=0.99)

    agent_a = tools.DQNAgent(obs_space_dims, action_space_dims, lr=1e-4, gamma=0.99)

    read_model_b_path = "models_v3/b/temp_1717957005_epoch_20000.pth"

    read_model_a_path = "models_v3/a/temp_1717957005_epoch_20000.pth"

    os.makedirs(f"outputs_v3/{current_time}/a", exist_ok=True)
    os.makedirs(f"outputs_v3/{current_time}/b", exist_ok=True)

    try:
        agent_a.load_model(read_model_a_path)
    except FileNotFoundError:
        print(f"No saved model a found at {read_model_a_path}. Starting from scratch.")

    try:
        agent_b.load_model(read_model_b_path)
    except FileNotFoundError:
        print(f"No saved model b found at {read_model_b_path}. Starting from scratch.")

    auto_save_epochs = 1000
    episode_interrupted = 0
    episode_of_best_reward = 0
    reward_list = []

    try:
        # create viewer
        # with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        total_num_episodes = int(200000)
        episode = 0

        # while viewer.is_running() and episode < total_num_episodes:
        while episode < total_num_episodes:
            done = False
            data.time = 0
            print('The episode is:', episode)

            # 重置环境到初始状态
            mujoco.mj_resetData(model, data)
            seed = random.randint(0, 10000)
            tools.random_state(data, seed)

            while not done:
                state = tools.get_obs_lifer(data)
                state_theta = np.arctan2(state[2], state[3])

                if abs(state_theta) < 0.23:
                    action = agent_a.sample_action(state)
                    data.ctrl[0] = action

                    mujoco.mj_step(model, data)

                    next_state = tools.get_obs_lifer(data)
                    done = data.time > 30 or abs(next_state[0]) > 1.45
                    reward = calculate_reward_a(next_state)

                    agent_a.store_transition(state, action, reward, next_state, done)
                    agent_a.update()

                    state = next_state
                else:
                    action = agent_b.sample_action(state)
                    data.ctrl[0] = action

                    mujoco.mj_step(model, data)

                    next_state = tools.get_obs_lifer(data)
                    done = data.time > 30 or abs(next_state[0]) > 1.25
                    reward = calculate_reward_b(next_state)

                    agent_b.store_transition(state, action, reward, next_state, done)
                    agent_b.update()

                    state = next_state

            # reward_episode = calculate_reward_b(tools.get_obs_lifer(data))
            # reward_list.append(reward_episode)
            # print(f"Episode {episode} ended with reward {reward_episode}.")

            episode += 1
            episode_interrupted += 1

            if episode % auto_save_epochs == 0:
                agent_a.save_model(f"outputs_v3/{current_time}/a/temp_{int(time.time())}_epoch_{episode}.pth")
                agent_b.save_model(f"outputs_v3/{current_time}/b/temp_{int(time.time())}_epoch_{episode}.pth")

            # if reward_episode > max reward in reward_list, save model
            # if reward_episode >= max(reward_list):
            #     episode_of_best_reward = episode
            #     agent.save_model(f"outputs/{current_time}/best_reward_temp_model.pth")

        agent_a.save_model(f"outputs_v3/{current_time}/a/model_{int(time.time())}_epoch_{episode}.pth")
        agent_b.save_model(f"outputs_v3/{current_time}/b/model_{int(time.time())}_epoch_{episode}.pth")
        print(f"Training completed. Models all saved.")
        # print(f"Best reward: {max(reward_list)} at episode {episode_of_best_reward}.")

        # plt.plot(reward_list)
        # plt.xlabel('Episode')
        # plt.ylabel('Reward')
        # plt.title('Reward Curve')
        # plt.savefig(f"outputs/{current_time}/reward_curve.png")
        # plt.show()

    except KeyboardInterrupt:
        agent_a.save_model(f"outputs_v3/{current_time}/a/temp_{int(time.time())}_epoch_{episode_interrupted}.pth")
        agent_b.save_model(f"outputs_v3/{current_time}/b/temp_{int(time.time())}_epoch_{episode_interrupted}.pth")
        print("Training interrupted. Model saved.")
        # print(f"Best reward: {max(reward_list)} at episode {episode_of_best_reward}.")

        # plt.plot(reward_list)
        # plt.xlabel('Episode')
        # plt.ylabel('Reward')
        # plt.title('Reward Curve')
        # plt.savefig(f"outputs/{current_time}/reward_curve.png")
        # plt.show()
