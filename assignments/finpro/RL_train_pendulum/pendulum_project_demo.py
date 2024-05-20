from __future__ import annotations

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

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

if __name__ == "__main__":
    env = gym.make("InvertedPendulum-v4", render_mode="human")  # Initialize the environment
    # env = gym.make("InvertedDoublePendulum-v4")
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Wrap the environment to record statistics
    
    obs_space_dims = env.observation_space.shape[0]  # Dimension of the observation space
    action_space_dims = env.action_space.shape[0]  # Dimension of the action space
    agent = Agent(obs_space_dims, action_space_dims)  # Instantiate the agent
    
    total_num_episodes = int(5e3)  # Total number of episodes
    
    # Simulation main loop
    for episode in range(total_num_episodes):
        obs, info = wrapped_env.reset()  # Reset the environment at the start of each episode
        done = False
        while not done:
            action = agent.sample_action(obs)  # Sample an action based on the current observation
            obs, reward, terminated, truncated, _ = wrapped_env.step(action)  # Take the action in the environment
            done = terminated or truncated  # Check if the episode has terminated
        # The collection of rewards and log probabilities should happen within the loop.
        # agent.update(rewards, log_probs)  # Update the policy based on the episode's experience
