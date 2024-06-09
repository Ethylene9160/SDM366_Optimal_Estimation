import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class DDPGAgent:
    """Agent that learns to solve tasks using the DDPG algorithm."""

    def __init__(self, obs_space_dims: int, action_space_dims: int, lr: float = 1e-3, gamma: float = 0.99,
                 tau: float = 1e-3):
        """Initializes the agent with actor and critic networks along with their target networks.

        Args:
            obs_space_dims (int): Dimension of the observation space.
            action_space_dims (int): Dimension of the action space.
            lr (float): Learning rate for the optimizer.
            gamma (float): Discount factor for future rewards.
            tau (float): Soft update parameter.
        """
        self.actor = ActorNetwork(obs_space_dims, action_space_dims)
        self.critic = CriticNetwork(obs_space_dims, action_space_dims)
        self.target_actor = ActorNetwork(obs_space_dims, action_space_dims)
        self.target_critic = CriticNetwork(obs_space_dims, action_space_dims)
        self.update_target_network(self.target_actor, self.actor, tau=1.0)
        self.update_target_network(self.target_critic, self.critic, tau=1.0)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau

    def select_action(self, state: np.ndarray, noise: float = 0.1) -> float:
        """Selects an action using the actor network and adds noise for exploration.

        Args:
            state (np.ndarray): The current state observation from the environment.
            noise (float): Noise scale for exploration.

        Returns:
            float: The action selected by the actor network.
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        action += noise * np.random.randn(action.shape[0])
        return np.clip(action, -15.0, 15.0).item()  # Ensure action is a scalar

    def update(self, replay_buffer, batch_size: int):
        """Updates the actor and critic networks using samples from the replay buffer.

        Args:
            replay_buffer: The replay buffer containing experience tuples.
            batch_size (int): Number of samples to use for updating the networks.
        """
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions).unsqueeze(1)  # Ensure actions have the correct shape
        rewards = torch.FloatTensor(rewards).unsqueeze(1)  # Ensure rewards have the correct shape
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)  # Ensure dones have the correct shape

        # Update Critic
        next_actions = self.target_actor(next_states)
        target_q_values = self.target_critic(next_states, next_actions)
        target_q_values = rewards + (1 - dones) * self.gamma * target_q_values
        q_values = self.critic(states, actions)

        critic_loss = nn.MSELoss()(q_values, target_q_values.detach())
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.optimizer_critic.step()

        # Update Actor
        policy_loss = -self.critic(states, self.actor(states)).mean()
        self.optimizer_actor.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        self.optimizer_actor.step()

        # Update target networks
        self.update_target_network(self.target_actor, self.actor)
        self.update_target_network(self.target_critic, self.critic)

    def update_target_network(self, target: nn.Module, source: nn.Module, tau: float = None):
        """Performs a soft update of the target network parameters.

        Args:
            target (nn.Module): The target network.
            source (nn.Module): The source network.
            tau (float, optional): The soft update parameter.
        """
        tau = self.tau if tau is None else tau
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save_model(self, path: str):
        """Saves the actor and critic networks to the specified path.

        Args:
            path (str): The path where the model will be saved.
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'optimizer_actor_state_dict': self.optimizer_actor.state_dict(),
            'optimizer_critic_state_dict': self.optimizer_critic.state_dict()
        }, path)
        print(f'Successfully saved model to {path}.')

    def load_model(self, path: str):
        """Loads the actor and critic networks from the specified path.

        Args:
            path (str): The path from where the model will be loaded.
        """
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
        self.actor.train()
        self.critic.train()
        print(f"Successfully loaded model from {path}.")


class ActorNetwork(nn.Module):
    """Neural network to parameterize the policy by predicting actions."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes layers of the neural network.

        Args:
            obs_space_dims (int): Dimension of the observation space.
            action_space_dims (int): Dimension of the action space.
        """
        super().__init__()
        self.fc1 = nn.Linear(obs_space_dims, 128)
        self.fc2 = nn.Linear(128, 200)
        self.fc3 = nn.Linear(200, action_space_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predicts actions given the state.

        Args:
            x (torch.Tensor): The state observation.

        Returns:
            torch.Tensor: Predicted action.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))
        return action


class CriticNetwork(nn.Module):
    """Neural network to estimate the Q-value of a given state-action pair."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes layers of the neural network.

        Args:
            obs_space_dims (int): Dimension of the observation space.
            action_space_dims (int): Dimension of the action space.
        """
        super().__init__()
        self.fc1 = nn.Linear(obs_space_dims + action_space_dims, 128)
        self.fc2 = nn.Linear(128, 200)
        self.fc3 = nn.Linear(200, 1)

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predicts Q-value given the state and action.

        Args:
            x (torch.Tensor): The state observation.
            action (torch.Tensor): The action taken.

        Returns:
            torch.Tensor: Predicted Q-value of the state-action pair.
        """
        x = torch.cat([x, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


class ReplayBuffer:
    """Replay buffer to store experience tuples."""

    def __init__(self, capacity: int):
        """Initializes the replay buffer.

        Args:
            capacity (int): Maximum number of experiences to store.
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Adds an experience tuple to the buffer.

        Args:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            next_state: The next state.
            done: Whether the episode is done.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        """Samples a batch of experiences from the buffer.

        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            tuple: Batch of states, actions, rewards, next states, and done flags.
        """
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in batch])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)
