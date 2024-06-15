from __future__ import annotations

import os
import time
import random

import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

import tools


# Environment reward function
def calculate_reward(state_list):
    """Calculate the reward based on the given state and done flag."""
    x, theta, sin_theta, cos_theta, x_dot, theta_dot = state_list

    # calculate theta in [-pi, pi] by sin and cos
    theta = np.arctan2(sin_theta, cos_theta)

    if abs(theta) >= 0.75:
        r = -(0.2 * x ** 2 + 0.6 * theta ** 2 - 0.005 * theta_dot ** 2)
        if abs(x) >= 1:
            r -= 10
    else:
        r = -(0.4 * x ** 2 + 0.6 * theta ** 2 + 0.001 * theta_dot ** 2)

    if abs(theta) >= 0.35 and (x_dot * theta) > 0:
        r -= 0.01 * x_dot ** 2
        if abs(x) >= 0.5:
            r -= 1
    return r



if __name__ == "__main__":
    xml_path = "Lifer_inverted_swing_pendulum.xml"
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    model, data = tools.init_mujoco(xml_path)

    obs_space_dims = 6
    action_space_dims = model.nu
    agent = tools.DQNAgent(obs_space_dims, action_space_dims, lr=3e-4, gamma=0.99)

    read_model_path = "models/temp_1717851834_epoch_454.pth"

    os.makedirs(f"outputs/{current_time}", exist_ok=True)

    try:
        agent.load_model(read_model_path)
    except FileNotFoundError:
        print(f"No saved model found at {read_model_path}. Starting from scratch.")

    auto_save_epochs = 100
    episode_interrupted = 0
    episode_of_best_reward = 0
    reward_list = []

    try:
        # create viewer
        # with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        total_num_episodes = int(20000)
        episode = 0

        # while viewer.is_running() and episode < total_num_episodes:
        while episode < total_num_episodes:
            done = False
            data.time = 0
            # print('The episode is:', episode)

            # 重置环境到初始状态
            mujoco.mj_resetData(model, data)
            seed = random.randint(0, 10000)
            tools.random_state(data, seed)

            while not done:
                state = tools.get_obs_lifer(data)
                action = agent.sample_action(state)
                data.ctrl[0] = action
                mujoco.mj_step(model, data)

                next_state = tools.get_obs_lifer(data)
                done = data.time > 45  # Example condition to end episode

                reward = calculate_reward(next_state)

                agent.store_transition(state, action, reward, next_state, done)
                agent.update()

                state = next_state

            reward_episode = calculate_reward(tools.get_obs_lifer(data))
            reward_list.append(reward_episode)
            print(f"Episode {episode} ended with reward {reward_episode}.")

            episode += 1
            episode_interrupted += 1

            if episode % auto_save_epochs == 0:
                agent.save_model(f"outputs/{current_time}/temp_{int(time.time())}_epoch_{episode}.pth")

            # if reward_episode > max reward in reward_list, save model
            if reward_episode >= max(reward_list):
                episode_of_best_reward = episode
                agent.save_model(f"outputs/{current_time}/best_reward_temp_model.pth")

        agent.save_model(f"outputs/{current_time}/model_{int(time.time())}_epoch_{episode}.pth")
        print(f"Training completed. Model saved.")
        print(f"Best reward: {max(reward_list)} at episode {episode_of_best_reward}.")

        plt.plot(reward_list)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward Curve')
        plt.savefig(f"outputs/{current_time}/reward_curve.png")
        # plt.show()

    except KeyboardInterrupt:
        agent.save_model(f"outputs/{current_time}/temp_{int(time.time())}_epoch_{episode_interrupted}.pth")
        print("Training interrupted. Model saved.")
        print(f"Best reward: {max(reward_list)} at episode {episode_of_best_reward}.")

        plt.plot(reward_list)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward Curve')
        plt.savefig(f"outputs/{current_time}/reward_curve.png")
        # plt.show()
