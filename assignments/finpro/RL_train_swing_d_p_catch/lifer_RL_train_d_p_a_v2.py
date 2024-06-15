from __future__ import annotations

import os
import time

from matplotlib import pyplot as plt
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env

import tools

if __name__ == "__main__":
    xml_path = "inverted_double_pendulum.xml"
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    model, data = tools.init_mujoco(xml_path)

    os.makedirs(f"outputs/v2/{current_time}", exist_ok=True)

    obs_space_dims = 10
    action_space_dims = model.nu

    read_model_path = "models/v2/temp_1718387515_steps_20000.pth"
    # read_model_path = ""

    env = tools.CustomEnv(model, data)
    check_env(env)

    try:
        agent = DDPG.load(read_model_path, env=env)
    except PermissionError:
        print(f"No saved model found at {read_model_path}. Starting from scratch.")
        agent = DDPG('MlpPolicy', env, verbose=1)

    total_time_steps = 20000
    episode_rewards = []


    class RewardCallback(BaseCallback):
        def __init__(self, env, verbose=0):
            super(RewardCallback, self).__init__(verbose)
            self.env = env
            self.newest_reward = 0

        def _on_step(self) -> bool:
            done = self.locals['dones']
            if done[0]:  # When episode ends
                episode_rewards.append(self.newest_reward)
            self.newest_reward = self.env.episode_reward
            return True


    reward_callback = RewardCallback(env)

    try:
        agent.learn(total_time_steps, callback=reward_callback)
        print("Training completed.")
        time_steps = agent.num_timesteps
        model_save_path = f"outputs/v2/{current_time}/temp_{int(time.time())}_steps_{time_steps}.pth"
        agent.save(model_save_path)
        print(f"Model saved at {model_save_path}")
    except KeyboardInterrupt:
        print("Training interrupted. Saving the model.")
        time_steps = agent.num_timesteps
        model_save_path = f"outputs/v2/{current_time}/temp_{int(time.time())}_steps_{time_steps}.pth"
        agent.save(model_save_path)
        print(f"Model saved at {model_save_path}")

    finally:
        # Plotting the rewards
        plt.figure(figsize=(10, 5))
        plt.plot(episode_rewards, label='Episode Reward')
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.title('Total Reward per Episode')
        plt.legend()
        plt.grid()
        plt.savefig(f"outputs/v2/{current_time}/training_rewards.png")
        plt.show()
