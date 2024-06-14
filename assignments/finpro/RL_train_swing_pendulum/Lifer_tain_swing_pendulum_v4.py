from __future__ import annotations

import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

import tools


# def save_reward_list(reward_list_a, reward_list_b, current_time):
#     # plot rewards by ppo
#     import matplotlib.pyplot as plt
#     plt.subplot(2, 1, 1)
#     plt.plot(reward_list_a, 'b')
#     plt.title('Reward by PPO agent a')
#     plt.xlabel('Episode')
#     plt.ylabel('Reward')
#
#     plt.subplot(2, 1, 2)
#     plt.plot(reward_list_b, 'b')
#     plt.title('Reward by PPO agent b')
#     plt.xlabel('Episode')
#     plt.ylabel('Reward')
#
#     plt.tight_layout()
#     plt.savefig(f"outputs_v4/{current_time}/reward_curve.png")


# Environment reward function


def calculate_reward_a(state_list, action):
    """Calculate the reward based on the given state."""
    x, theta, sin_theta, cos_theta, x_dot, theta_dot = state_list

    # calculate theta in [-pi, pi] by sin and cos
    theta = np.arctan2(sin_theta, cos_theta)

    r = -(0.8 * theta ** 2 + 0.1 * x ** 2 + 0.001 * theta_dot ** 2 + 0.1 * x_dot ** 2)
    r += 1.0 if abs(x) < 1.25 else 0.0

    return r


# Environment reward function
def calculate_reward_b(state_list, action):
    """Calculate the reward based on the given state."""
    x, theta, sin_theta, cos_theta, x_dot, theta_dot = state_list

    # calculate theta in [-pi, pi] by sin and cos
    theta = np.arctan2(sin_theta, cos_theta)

    r = -(0.1 * x ** 2 + 1 * theta ** 2 + 0.001 * action ** 2)

    if abs(theta) < 0.30:
        r += 1

    if abs(x) < 1.25 and abs(theta) < 0.30 and abs(x_dot) < 2.0 and abs(theta_dot) < 2.0:
        r += 1

    # if abs(x) >= 1.5:
    #     r -= 1

    return r


if __name__ == "__main__":
    xml_path = "Lifer_inverted_swing_pendulum.xml"
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    model, data = tools.init_mujoco(xml_path)

    os.makedirs(f"outputs_v4/{current_time}/a", exist_ok=True)
    os.makedirs(f"outputs_v4/{current_time}/b", exist_ok=True)

    total_num_episodes = 1000000

    training_a = 1

    if training_a:
        read_model_a_path = "models_v4/a/temp_1718342216_epoch_1000000.lifer"
        # read_model_a_path = ""

        env_a = tools.CustomEnv_a(model, data, calculate_reward_a)

        check_env(env_a)

        # agent_a = PPO('MlpPolicy', env_a, verbose=1)

        try:
            agent_a = PPO.load(read_model_a_path, env=env_a)
        except PermissionError:
            agent_a = PPO('MlpPolicy', env_a, verbose=1)
            print(f"No saved model a found at {read_model_a_path}. Starting from scratch.")

        agent_a.learn(total_timesteps=total_num_episodes)
        model_save_path = f"outputs_v4/{current_time}/a/temp_{int(time.time())}_epoch_{total_num_episodes}.zip"
        agent_a.save(model_save_path)
        print(f"Training finished. Model saved at '{model_save_path}'")

    else:

        read_model_b_path = "models_v4/b/temp_1718311298_epoch_2000000.lifer"
        # read_model_b_path = ""

        env_b = tools.CustomEnv_b(model, data, calculate_reward_b)

        check_env(env_b)

        # agent_b = PPO('MlpPolicy', env_b, verbose=1)

        try:
            agent_b = PPO.load(read_model_b_path, env=env_b)
        except PermissionError:
            agent_b = PPO('MlpPolicy', env_b, verbose=1)
            print(f"No saved model b found at {read_model_b_path}. Starting from scratch.")

        agent_b.learn(total_timesteps=total_num_episodes)

        model_save_path = f"outputs_v4/{current_time}/b/temp_{int(time.time())}_epoch_{total_num_episodes}.zip"
        agent_b.save(model_save_path)
        print(f"Training finished. Model saved at '{model_save_path}'")

    exit(0)

    # auto_save_epochs = 1000
    # episode_interrupted = 0
    # reward_list_a = []
    # reward_list_b = []
    #
    # try:
    #     for episode in range(1, total_num_episodes + 1):
    #         state = env_a.reset()
    #         done = False
    #         while not done:
    #             state_theta = np.arctan2(state[2], state[3])
    #
    #             if abs(state_theta) < 0.23:
    #                 action, _ = agent_a.predict(state)
    #                 action_value = action[0]
    #                 state, reward, done, _ = env_a.step(action_value)
    #
    #                 if done:
    #                     reward_list_a.append(reward)
    #                     print(
    #                         f"Episode {episode} completed in a with reward {reward}, theta = {state_theta}, x = {state[0]}")
    #
    #             else:
    #                 action, _ = agent_b.predict(state)
    #                 action_value = action[0]
    #                 state, reward, done, _ = env_b.step(action_value)
    #
    #                 if done:
    #                     reward_list_b.append(reward)
    #                     print(
    #                         f"Episode {episode} completed in b with reward {reward}, theta = {state_theta}, x = {state[0]}")
    #
    #         if episode % auto_save_epochs == 0:
    #             agent_a.save(f"outputs_v4/{current_time}/a/temp_{int(time.time())}_epoch_{episode}.zip")
    #             agent_b.save(f"outputs_v4/{current_time}/b/temp_{int(time.time())}_epoch_{episode}.zip")
    #             save_reward_list(reward_list_a, reward_list_b, current_time)
    #
    #         # print(f"Episode {episode} completed.")
    #         episode_interrupted += 1
    #
    #     agent_a.save(f"outputs_v4/{current_time}/a/model_{int(time.time())}_epoch_{total_num_episodes}.zip")
    #     agent_b.save(f"outputs_v4/{current_time}/b/model_{int(time.time())}_epoch_{total_num_episodes}.zip")
    #     print(f"Training completed. Models all saved.")
    #
    #     save_reward_list(reward_list_a, reward_list_b, current_time)
    #     print("Reward curve saved.")
    #
    # except KeyboardInterrupt:
    #     agent_a.save(f"outputs_v4/{current_time}/a/temp_{int(time.time())}_epoch_{episode_interrupted}.zip")
    #     agent_b.save(f"outputs_v4/{current_time}/b/temp_{int(time.time())}_epoch_{episode_interrupted}.zip")
    #     print("Training interrupted. Models saved.")
    #
    #     save_reward_list(reward_list_a, reward_list_b, current_time)
    #     print("Reward curve saved.")
