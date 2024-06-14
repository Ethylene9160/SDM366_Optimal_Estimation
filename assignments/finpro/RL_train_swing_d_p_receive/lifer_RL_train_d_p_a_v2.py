from __future__ import annotations

import os
import time

from matplotlib import pyplot as plt
from stable_baselines3 import DDPG
from stable_baselines3.common.env_checker import check_env

import tools


def plot_rewards(rewards, folder_name, episode_record=0):
    plt.figure()
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward with Episode')
    plt.savefig(f'{folder_name}/episode_{episode_record}_rewards.eps', format='eps')
    plt.show()


if __name__ == "__main__":
    xml_path = "inverted_double_pendulum.xml"
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    model, data = tools.init_mujoco(xml_path)

    os.makedirs(f"outputs_v2/{current_time}", exist_ok=True)

    obs_space_dims = 10
    action_space_dims = model.nu

    read_model_path = "models_v2/"
    read_model_path = ""

    env = tools.CustomEnv(model, data)
    check_env(env)

    try:
        agent = DDPG.load(read_model_path, env=env)
    except PermissionError:
        print(f"No saved model found at {read_model_path}. Starting from scratch.")
        agent = DDPG('MlpPolicy', env, verbose=1)

    total_time_steps = 20000

    try:
        agent.learn(total_time_steps)
        print("Training completed.")
        model_save_path = f"outputs_v2/{current_time}/temp_{time.time()}_steps_{total_time_steps}.pth"
        agent.save(model_save_path)
        print(f"Model saved at {model_save_path}")
    except KeyboardInterrupt:
        print("Training interrupted. Saving the model.")
        model_save_path = f"outputs_v2/{current_time}/temp_{time.time()}_steps_{total_time_steps}.pth"
        agent.save(model_save_path)
        print(f"Model saved at {model_save_path}")
