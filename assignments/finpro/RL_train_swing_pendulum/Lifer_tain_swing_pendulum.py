from __future__ import annotations

import os
import time

import mujoco.viewer

import tools


# Environment reward function
def calculate_reward(state_list):
    """Calculate the reward based on the given state and done flag."""
    x, theta, theta_dot = state_list[0], state_list[1], state_list[5]
    if abs(theta) >= 0.5:
        r = -(1.5 * x ** 2 + 0.6 * theta ** 2 - 0.005 * theta_dot ** 2)
    else:
        r = -(0.4 * x ** 2 + 0.6 * theta ** 2 + 0.001 * theta_dot ** 2)
    bond = abs(x) >= 0.5
    if bond:
        r -= 1000
    return r


if __name__ == "__main__":
    xml_path = "Lifer_inverted_swing_pendulum.xml"
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    model, data = tools.init_mujoco(xml_path)
    os.makedirs(f"{current_time}", exist_ok=True)
    # print(model.actuator_ctrlrange)

    obs_space_dims = 6
    action_space_dims = model.nu
    agent = tools.DQNAgent(obs_space_dims, action_space_dims, lr=3e-4, gamma=0.99)

    read_model_path = "models/temp_model_save_at_epoch_100.pth"
    save_model_path = "dqn_policy_v0.pth"
    try:
        agent.load_model(read_model_path)
    except FileNotFoundError:
        print(f"No saved model found at {read_model_path}. Starting from scratch.")

    auto_save_epochs = 100
    episode_interrupted = 0

    try:
        # create viewer
        # with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        total_num_episodes = int(20000)
        episode = 0
        # while viewer.is_running() and episode < total_num_episodes:
        while episode < total_num_episodes:
            done = False
            data.time = 0
            print('The episode is:', episode)

            # 重置环境到初始状态
            mujoco.mj_resetData(model, data)
            tools.random_state(data)

            while not done:
                state = tools.get_obs_lifer(data)
                action = agent.sample_action(state)
                data.ctrl[0] = action
                mujoco.mj_step(model, data)

                next_state = tools.get_obs_lifer(data)
                # print(next_state)
                done = data.time > 45  # Example condition to end episode

                reward = calculate_reward(next_state)
                agent.store_transition(state, action, reward, next_state, done)
                agent.update()

                state = next_state

            episode += 1
            episode_interrupted += 1

            if episode % auto_save_epochs == 0:
                agent.save_model(f"{current_time}/temp_model_save_at_epoch_{episode}.pth")

        agent.save_model(f"{current_time}/{save_model_path}")

    except KeyboardInterrupt:
        agent.save_model(f"{current_time}/autosave_at_epoch_{episode_interrupted}.pth")
        print("Training interrupted. Model saved.")
