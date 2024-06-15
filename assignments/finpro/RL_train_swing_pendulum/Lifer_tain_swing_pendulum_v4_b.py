from __future__ import annotations

import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

import tools


# Environment reward function
def calculate_reward_b(state_list, action):
    """Calculate the reward based on the given state."""
    x, theta, sin_theta, cos_theta, x_dot, theta_dot = state_list

    # calculate theta in [-pi, pi] by sin and cos
    theta = np.arctan2(sin_theta, cos_theta)

    r = -(0.1 * x ** 2 + 1 * theta ** 2 + 0.001 * theta_dot ** 2 + 0.1 * x_dot ** 2 + 0.001 * action ** 2)
    # r = 0

    if abs(theta) < 0.30:
        r += 1

    # if abs(x) < 1.25 and abs(theta) < 0.30 and abs(x_dot) < 2.0 and abs(theta_dot) < 2.0:
    #     r += 1

    # if abs(x) >= 1.5:
    #     r -= 1

    return r


if __name__ == "__main__":
    xml_path = "Lifer_inverted_swing_pendulum.xml"
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    model, data = tools.init_mujoco(xml_path)

    os.makedirs(f"outputs_v4/{current_time}/a", exist_ok=True)
    os.makedirs(f"outputs_v4/{current_time}/b", exist_ok=True)

    total_num_episodes = 2000000

    read_model_b_path = "models_v4/b/temp_1718311298_epoch_2000000.lifer"
    read_model_b_path = ""

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
