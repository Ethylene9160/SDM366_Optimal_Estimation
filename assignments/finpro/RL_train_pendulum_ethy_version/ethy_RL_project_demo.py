from __future__ import annotations

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import mujoco
import glfw


class Agent:
    """Agent that learns to solve the Inverted Pendulum task using a policy gradient algorithm.
    The agent utilizes a policy network to sample actions and update its policy based on
    collected rewards.
    """

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes the agent with a neural network policy.

        Args:
            obs_space_dims (int): Dimension of the observation space.
            action_space_dims (int): Dimension of the action space.
        """
        self.policy_network = Policy_Network(obs_space_dims, action_space_dims)

    def sample_action(self, state: np.ndarray) -> float:
        """Samples an action according to the policy network given the current state.

        Args:
            state (np.ndarray): The current state observation from the environment.

        Returns:
            float: The action sampled from the policy distribution.
        """
        return np.array([0])  # Return the action

    def update(self, rewards, log_probs):
        """Updates the policy network using the REINFORCE algorithm based on collected rewards and log probabilities.

        Args:
            rewards (list): Collected rewards from the environment.
            log_probs (list): Log probabilities of the actions taken.
        """
        # The actual implementation of the REINFORCE update will be done here.
        pass


class Policy_Network(nn.Module):
    """Neural network to parameterize the policy by predicting action distribution parameters."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes layers of the neural network.

        Args:
            obs_space_dims (int): Dimension of the observation space.
            action_space_dims (int): Dimension of the action space.
        """
        super().__init__()
        # Define the neural network layers here
        pass

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predicts parameters of the action distribution given the state.

        Args:
            x (torch.Tensor): The state observation.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Predicted mean and standard deviation of the action distribution.
        """
        # Implement the prediction logic here
        return torch.tensor(0.0), torch.tensor(1.0)  # Example placeholders


def init_mujoco(model_path):
    mujoco_model = mujoco.MjModel.from_xml_path(model_path)
    mujoco_data = mujoco.MjData(mujoco_model)
    return mujoco_model, mujoco_data


def init_glfw(width=640, height=480):
    if not glfw.init():
        return None
    window = glfw.create_window(width, height, "MuJoCo Simulation", None, None)
    if not window:
        glfw.terminate()
        return None
    glfw.make_context_current(window)
    return window


def render(window, mujoco_model, mujoco_data):
    while not glfw.window_should_close(window):
        mujoco.mj_step(mujoco_model, mujoco_data)

        viewport = mujoco.MjrRect(0, 0, 640, 480)
        scene = mujoco.MjvScene(mujoco_model, maxgeom=1000)
        context = mujoco.MjrContext(mujoco_model, mujoco.mjtFontScale.mjFONTSCALE_150)

        mujoco.mjv_updateScene(mujoco_model, mujoco_data, mujoco.MjvOption(), None, mujoco.MjvCamera(),
                               mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(viewport, scene, context)

        glfw.swap_buffers(window)
        glfw.poll_events()
    glfw.terminate()


if __name__ == "__main__":
    xml_path = "inverted_pendulum.xml"

    model, data = init_mujoco(xml_path)
    window = init_glfw()

    if window:
        obs_space_dims = model.nq
        action_space_dims = model.nu
        agent = Agent(obs_space_dims, action_space_dims)

        total_num_episodes = int(5e3)

        for episode in range(total_num_episodes):
            mujoco.mj_step(model, data)
            render(window, model, data)

        glfw.terminate()
