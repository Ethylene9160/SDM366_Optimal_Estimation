from __future__ import annotations

import os
import pickle

import numpy as np
import mujoco
import mujoco.viewer
import time
import random

import torch


class QLearningAgent:
    """Agent that learns to solve the Inverted Pendulum task using Q-learning."""

    def __init__(self, obs_space_dims: int, action_space: list[float], alpha: float = 0.1, gamma: float = 0.99,
                 epsilon: float = 0.1):
        """Initializes the agent with a Q-table and hyperparameters.

        Args:
            obs_space_dims (int): Dimension of the observation space.
            action_space (list[float]): List of possible actions.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Exploration rate.
        """
        self.q_table = {}
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.init_action = -15.0

    def get_discrete_state(self, state: np.ndarray) -> tuple:
        """Converts the continuous state to a discrete state for Q-table indexing.

        Args:
            state (np.ndarray): The continuous state observation.

        Returns:
            tuple: The discretized state.
        """
        return tuple(np.round(state, 2))  # Example discretization, modify as needed

    def sample_action(self, state: np.ndarray) -> float:
        """Samples an action using an epsilon-greedy policy.

        Args:
            state (np.ndarray): The current state observation from the environment.

        Returns:
            float: The action selected from the action space.
        """
        discrete_state = self.get_discrete_state(state)
        if random.uniform(0, 1) < self.epsilon or discrete_state not in self.q_table:
            return random.choice(self.action_space)
        else:
            return max(self.q_table[discrete_state], key=self.q_table[discrete_state].get)

    def update(self, state: np.ndarray, action: float, reward: float, next_state: np.ndarray):
        """Updates the Q-table based on the agent's experience.

        Args:
            state (np.ndarray): The current state.
            action (float): The action taken.
            reward (float): The reward received.
            next_state (np.ndarray): The next state observed.
        """
        discrete_state = self.get_discrete_state(state)
        discrete_next_state = self.get_discrete_state(next_state)

        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = {a: 0 for a in self.action_space}

        if discrete_next_state not in self.q_table:
            self.q_table[discrete_next_state] = {a: 0 for a in self.action_space}

        best_next_action = max(self.q_table[discrete_next_state], key=self.q_table[discrete_next_state].get)
        td_target = reward + self.gamma * self.q_table[discrete_next_state][best_next_action]
        td_error = td_target - self.q_table[discrete_state][action]

        self.q_table[discrete_state][action] += self.alpha * td_error

    def save_model(self, path: str):
        """Saves the Q-table to a file using pickle.

        Args:
            path (str): The path to save the Q-table.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved to {path}")

    def load_model(self, path: str):
        """Loads the Q-table from a file using pickle.

        Args:
            path (str): The path to load the Q-table from.
        """
        with open(path, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"Q-table loaded from {path}")
