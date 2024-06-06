from __future__ import annotations

import os
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import mujoco
import mujoco.viewer
import time

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

if __name__ == "__main__":
    xml_path = "inverted_pendulum.xml"

    model, data = init_mujoco(xml_path)

    obs_space_dims = model.nq
    action_space_dims = model.nu
    agent = Agent(obs_space_dims, action_space_dims)

    # create viewer
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        total_num_episodes = int(3e4)
        episode = 0
        while viewer.is_running() and episode < total_num_episodes:
            rewards = []
            log_probs = []
            states = []
            done = False
            data.time = 0

            # reset to initial state
            mujoco.mj_resetData(model, data)

            # TODO: Here, you need to add noise to the model manully!

            while not done:
                step_start = time.time()
                state = data.qpos
                action= agent.sample_action(state)
                data.ctrl[0] = action
                mujoco.mj_step(model, data)

                ######  Calculate the Reward. #######
                reward = 1.0
                ###### End. The same as the official model. ########

                rewards.append(reward)
                states.append(state.copy())
                done = data.time > 45 or abs(data.qpos[1]) > 0.18  # Example condition to end episode

                ####################################
                ### commit the following line to speed up the training, this will not show the simulation vedio frame.
                ####################################
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
                viewer.sync()
                
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                ####################################
                ####################################

            agent.update(rewards, log_probs)
            episode += 1

