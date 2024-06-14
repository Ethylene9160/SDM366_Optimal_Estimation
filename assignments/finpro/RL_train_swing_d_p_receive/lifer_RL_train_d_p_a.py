from __future__ import annotations

import os
# import glfw
import time

import mujoco.viewer
import numpy as np
from matplotlib import pyplot as plt

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

    os.makedirs(f"outputs/{current_time}", exist_ok=True)

    obs_space_dims = 10
    action_space_dims = model.nu
    agent = tools.A2C.A2CAgent(obs_space_dims, action_space_dims, lr=2e-4, gamma=0.99)

    read_model_path = "models/"
    read_model_path = ""

    try:
        agent.load_model(read_model_path)
    except FileNotFoundError:
        print(f"No saved model found at {read_model_path}. Starting from scratch.")

    total_num_episodes = int(20000)
    auto_save_epochs = 1000

    total_reward = []
    episode_record = 0

    try:
        i = 0
        episode = 0
        # create viewer
        # with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:

        # while viewer.is_running() and episode < total_num_episodes:
        while episode < total_num_episodes:
            rewards = []
            log_probs = []
            states = []

            done = False
            data.time = 0

            # 重置环境到初始状态
            mujoco.mj_resetData(model, data)
            seed = np.random.randint(0, 100000)
            tools.random_state(data, seed)
            state = tools.get_obs_lifer(data)

            while not done:
                action, log_prob = agent.sample_action(state)
                data.ctrl[0] = action

                mujoco.mj_step(model, data)
                state = tools.get_obs_lifer(data)

                # Calculate the Reward
                reward = 1.

                # The same as the official model
                p = 0.85 * data.qpos[0] ** 2 + 0.005 * data.qvel[0] ** 2 + 0.05 * data.qvel[1] ** 2
                reward -= p

                rewards.append(reward)
                log_probs.append(log_prob)
                states.append(state.copy())
                done = data.time > 45 or abs(data.qpos[2]) > 0.75

                agent.update(rewards, log_probs, states)

            i += 1
            if i == 50:
                i = 0
                total_reward.append(np.sum(np.array(rewards)))

            episode += 1
            episode_record += 1

            print(f"Episode {episode} lasted for {data.time:.2f} s. Total reward: {np.sum(np.array(rewards)):.2f}")
            if episode % auto_save_epochs == 0:
                model_save_path = f"outputs/{current_time}/temp_{int(time.time())}_epoch_{total_num_episodes}.pth"
                agent.save_model(model_save_path)

                if total_reward:
                    plot_file = f'outputs/{current_time}'
                    plot_rewards(total_reward, plot_file, episode_record)

        model_save_path = f"outputs/{current_time}/temp_{int(time.time())}_epoch_{total_num_episodes}.pth"
        agent.save_model(model_save_path)
        print(f"Training finished. Model saved at '{model_save_path}'")

        if total_reward:
            plot_file = f'outputs/{current_time}'
            plot_rewards(total_reward, plot_file, episode_record)

    except KeyboardInterrupt:
        model_save_path = f"outputs/{current_time}/temp_{int(time.time())}_epoch_{episode_record}.pth"
        agent.save_model(model_save_path)
        print("Training interrupted. Model saved.")

        if total_reward:
            plot_file = f'outputs/{current_time}'
            plot_rewards(total_reward, plot_file, episode_record)
