import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gpytorch
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, MultitaskKernel
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from torch.distributions import Normal
from gpytorch.settings import cholesky_jitter
import torch

def current_reward(state, xlim=0.85):
    '''
    state[0]: position of cart
    state[1]: theta
    state[2]: v of the cart
    state[3]: omega
    '''
    reward =-1.8 * state[1] ** 2 -  0.002 * state[3] ** 2 - 0.2* state[0] ** 2
    if abs(state[0]) > xlim:
        reward = -20.0
    elif abs(state[1]) < 0.20:
        reward += 3.5
    return reward

class PolicyNetwork(nn.Module):
    def __init__(self, obs_space_dims, action_space_dims):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_space_dims, 128)
        self.fc2 = nn.Linear(128, 64)
        self.mean = nn.Linear(64, action_space_dims)
        self.log_std = nn.Linear(64, action_space_dims)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        std = torch.exp(log_std)
        return mean, std


class MultitaskGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = RBFKernel()
        self.task_covar_module = MultitaskKernel(self.covar_module, num_tasks=train_y.size(-1), rank=1)

    def forward(self, x):
        mean_x = self.mean_module(x).unsqueeze(-1).repeat(1, self.task_covar_module.num_tasks)
        covar_x = self.task_covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)

class PILCOAgent:
    def __init__(self, obs_space_dims: int, action_space_dims: int, lr: float = 1e-3, gamma: float = 0.95):
        self.policy_network = PolicyNetwork(obs_space_dims, action_space_dims)
        self.optimizer_policy = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.gamma = gamma

        self.likelihood = GaussianLikelihood()
        self.gp_model = None

    def sample_action(self, state: np.ndarray) -> tuple[float, torch.Tensor]:
        state = torch.FloatTensor(state)
        mean, std = self.policy_network(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def create_gp_model(self, train_x, train_y):
        if self.gp_model is None:
            self.train_x = train_x
            self.train_y = train_y
            self.likelihood = MultitaskGaussianLikelihood(num_tasks=train_y.size(-1))
            self.gp_model = MultitaskGPModel(self.train_x, self.train_y, self.likelihood)
        else:
            self.train_x = train_x
            self.train_y = train_y
            self.gp_model.set_train_data(inputs=self.train_x, targets=self.train_y, strict=False)

    def train_gp_model(self, learning_rate=0.01, epochs=100):
        self.gp_model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.gp_model.parameters(), lr=learning_rate)
        mll = ExactMarginalLogLikelihood(self.likelihood, self.gp_model)
        for i in range(epochs):
            optimizer.zero_grad()
            output = self.gp_model(self.train_x)
            # print('shape of the output mean:', output.mean.shape)  # 应该是 (601, 4)
            # print('shape of the train_y:', self.train_y.shape)  # 应该是 (601, 4)
            loss = -mll(output, self.train_y)
            loss.backward()
            optimizer.step()

    def update(self, rewards, log_probs, states, actions, next_states):
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        discounted_rewards = torch.FloatTensor(discounted_rewards).unsqueeze(1)  # 确保 discounted_rewards 的形状为 [601, 1]

        log_probs = torch.stack(log_probs)
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions)).unsqueeze(1)  # 确保 actions 维度正确
        next_states = torch.FloatTensor(np.array(next_states))

        # 预测状态差值 delta_states
        delta_states = next_states - states  # delta_states 的维度应该是 (601, 6)

        # 确保 train_x 和 train_y 维度匹配
        train_x = torch.cat([states, actions], dim=-1)  # train_x 的维度是 (601, 7)
        train_y = delta_states  # 预测 delta_states，维度是 (601, 6)

        self.create_gp_model(train_x, train_y)
        self.train_gp_model()

        predicted_deltas = self.predict_next_state(states, actions).squeeze()
        predicted_next_states = states + predicted_deltas

        # 使用 current_reward 函数计算 predicted_next_states 的奖励
        predicted_rewards = torch.FloatTensor([current_reward(state) for state in predicted_next_states])

        # print('shape of discounted rewards: ', discounted_rewards.shape)  # 应该是 (601, 1)
        # print('shape of predicted rewards: ', predicted_rewards.shape)  # 应该是 (601, 1)

        advantages = discounted_rewards - predicted_rewards.unsqueeze(1)  # 确保 predicted_rewards 的形状为 [601, 1]

        # 更新策略网络
        policy_loss = -torch.sum(log_probs * advantages.detach())
        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        self.optimizer_policy.step()

    def predict_next_state(self, states, actions):
        self.gp_model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            state_action = torch.cat([states, actions], dim=-1)
            observed_pred = self.likelihood(self.gp_model(state_action))
            mean = observed_pred.mean
        return mean

    def save_model(self, path: str):
        torch.save({
            'policy_network_state_dict': self.policy_network.state_dict(),
            'optimizer_policy_state_dict': self.optimizer_policy.state_dict(),
            'gp_model_state_dict': self.gp_model.state_dict() if self.gp_model else None,
            'train_x': self.train_x,
            'train_y': self.train_y
        }, path)
        print(f'successfully save model to {path}.')

    def load_model(self, path: str):
        checkpoint = torch.load(path)
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.optimizer_policy.load_state_dict(checkpoint['optimizer_policy_state_dict'])
        if checkpoint['gp_model_state_dict']:
            self.create_gp_model(checkpoint['train_x'], checkpoint['train_y'])
            self.gp_model.load_state_dict(checkpoint['gp_model_state_dict'])
        self.policy_network.train()
        self.gp_model.train() if self.gp_model else None
        print(f"successfully load model from {path}.")

def preprocess_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / (std + 1e-8), mean, std

def denormalize_data(data, mean, std):
    return data * std + mean
