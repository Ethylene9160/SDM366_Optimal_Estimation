import concurrent.futures
import os
import time
import copy

import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal


class Agent:
    """Agent that learns to solve the Inverted Pendulum task using a policy gradient algorithm.
    The agent utilizes a policy network to sample actions and update its policy based on
    collected rewards.
    """

    def __init__(self, obs_space_dims: int, action_space_dims: int, lr: float = 1e-3):
        """Initializes the agent with a neural network policy.

        Args:
            obs_space_dims (int): Dimension of the observation space.
            action_space_dims (int): Dimension of the action space.
            lr (float): Learning rate for the optimizer.
        """
        self.policy_network = Policy_Network(obs_space_dims, action_space_dims)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.gamma = 0.75  # Discount factor

    def sample_action(self, state: np.ndarray) -> tuple[float, torch.Tensor]:
        """Samples an action according to the policy network given the current state.

        Args:
            state (np.ndarray): The current state observation from the environment.

        Returns:
            tuple[float, torch.Tensor]: The action sampled from the policy distribution and its log probability.
        """
        state = torch.FloatTensor(state)
        mean, std = self.policy_network(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()  # Sum over action dimensions
        return action.item(), log_prob

    def update(self, rewards: list, log_probs: list):
        """Updates the policy network using the REINFORCE algorithm based on collected rewards and log probabilities.

        Args:
            rewards (list): Collected rewards from the environment.
            log_probs (list): Log probabilities of the actions taken.
        """
        discounted_rewards = []
        cumulative_reward = 0
        # Reverse iterate over rewards
        for reward in reversed(rewards):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)

        discounted_rewards = torch.FloatTensor(discounted_rewards)
        log_probs = torch.stack(log_probs)

        # discounted_rewards = discounted_rewards.detach()

        loss = -torch.sum(log_probs * discounted_rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def save_model(model: nn.Module, path: str):
    """Saves the policy network to the specified path."""
    torch.save(model.state_dict(), path)
    print(f"Model saved at {path}")


def load_model(model: nn.Module, path: str):
    """Loads the policy network from the specified path."""
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        model.eval()
    else:
        print(f"No model found at {path}")


class Policy_Network(nn.Module):
    """Neural network to parameterize the policy by predicting action distribution parameters."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes layers of the neural network.

        Args:
            obs_space_dims (int): Dimension of the observation space.
            action_space_dims (int): Dimension of the action space.
        """
        super().__init__()
        self.fc1 = nn.Linear(obs_space_dims, 128)
        self.fc2 = nn.Linear(128, 64)
        self.mean = nn.Linear(64, action_space_dims)
        self.log_std = nn.Linear(64, action_space_dims)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predicts parameters of the action distribution given the state.

        Args:
            x (torch.Tensor): The state observation.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Predicted mean and standard deviation of the action distribution.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        std = torch.exp(log_std)
        return mean, std


def init_mujoco(model_path):
    mujoco_model = mujoco.MjModel.from_xml_path(model_path)
    mujoco_data = mujoco.MjData(mujoco_model)
    return mujoco_model, mujoco_data


# def simulate_episode(agent, model, data, render=False):
#     rewards = []
#     log_probs = []
#     done = False
#     data.time = 0
#     mujoco.mj_resetData(model, data)
#
#     while not done:
#         state = data.qpos
#         action, log_prob = agent.sample_action(state)
#         data.ctrl[0] = action
#         mujoco.mj_step(model, data)
#         reward = - (data.qpos[1] ** 2 + 0.1 * data.qvel[1] ** 2 + 0.001 * action ** 2)
#         rewards.append(reward)
#         log_probs.append(log_prob)
#         done = data.time > 3
#
#     return rewards, log_probs


def init_mujoco_thread_safe(model_path):
    """Initialize MuJoCo model and data for each thread to avoid shared resource conflicts."""
    mujoco_model = mujoco.MjModel.from_xml_path(model_path)
    mujoco_data = mujoco.MjData(mujoco_model)
    return mujoco_model, mujoco_data


def simulate_episode(agent_params, xml_path, model_path, render=False):
    model, data = init_mujoco_thread_safe(xml_path)
    agent = Agent(*agent_params)  # 每个线程创建自己的 Agent 实例

    load_model(agent.policy_network, model_path)  # 确保每个线程加载独立的模型

    rewards = []
    log_probs = []
    done = False
    data.time = 0
    mujoco.mj_resetData(model, data)

    while not done:
        state = data.qpos
        action, log_prob = agent.sample_action(state)
        data.ctrl[0] = action
        mujoco.mj_step(model, data)
        reward = - (data.qpos[1] ** 2 * data.qvel[1] ** 2)
        rewards.append(reward)
        log_probs.append(log_prob)
        done = data.time > 20

    agent.update(rewards, log_probs)  # 在每个线程内部更新 Agent
    return [param.grad for param in agent.policy_network.parameters()]


if __name__ == "__main__":
    xml_path = "inverted_pendulum.xml"
    model, data = init_mujoco(xml_path)

    # model.opt.timestep = 0.01  # 设置更大的时间步长
    # model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER  # 使用简单的欧拉积分器
    # model.opt.collision = 0  # 禁用碰撞检测
    # model.opt.cone = 0  # 禁用摩擦锥

    obs_space_dims = model.nq
    action_space_dims = model.nu
    agent_params = (obs_space_dims, action_space_dims)

    model_path = "ethy_official_modela.pth"
    save_model_path = "lifer_official_model0.pth"

    agent = Agent(*agent_params)
    load_model(agent.policy_network, model_path)

    total_num_episodes = int(5e3)
    episode = 0

    visualize = True  # 控制是否可视化展示训练成果
    visualize = False

    if visualize:
        # 使用MuJoCo的viewer进行可视化
        agent = Agent(*agent_params)
        load_model(agent.policy_network, save_model_path)

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and episode < total_num_episodes:
                # 重置环境到初始状态
                mujoco.mj_resetData(model, data)
                done = False

                while not done:
                    step_start = time.time()
                    state = data.qpos
                    action, _ = agent.sample_action(state)
                    data.ctrl[0] = action
                    mujoco.mj_step(model, data)
                    done = data.time > 20

                    with viewer.lock():
                        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
                    viewer.sync()

                    time_until_next_step = model.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)

                print(f"Episode {episode} displayed.")
                episode += 1
    else:
        # 后台运行仿真
        print(f"Starting simulation with {total_num_episodes} episodes...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(simulate_episode, agent_params, xml_path, model_path, visualize) for _ in
                       range(total_num_episodes)]

            for future in concurrent.futures.as_completed(futures):
                gradients = future.result()
                for param, grad in zip(agent.policy_network.parameters(), gradients):
                    if grad is not None:
                        if param.grad is None:
                            param.grad = grad
                        else:
                            param.grad += grad
                episode += 1
                print(f"Episode {episode}: Gradients aggregated")

            agent.optimizer.step()
            agent.optimizer.zero_grad()

        print("Simulation complete.")

    save_model(agent.policy_network, save_model_path)
