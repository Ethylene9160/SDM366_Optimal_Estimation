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
# torch.autograd.set_detect_anomaly(True)

def current_reward(state, xlim=0.95):
    '''
    state[0]: x
    state[1]: theta1
    state[2]: theta2
    state[3]: dx
    state[4]: dtheta1
    state[5]: dtheta2
    '''
    # y = 0.6*np.cos(state[1]) + 0.6*np.cos(state[1]+state[2])
    # return y
    t2 = state[1] + state[2]
    r = 5*np.cos(state[1]+np.cos(t2))-state[1]**2 - (t2)**2 - 0.01*state[0]**2 - 0.001*state[5]**2 - 0.0003*state[4]**2
    # if abs(state[0]) > xlim:
    #     r -= 10.0
    # if state[0] > 1.0:
    #     r += 10.0
    # r=-1
    return r


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
        # self.covar_module = gpytorch.kernels.ScaleKernel(
        #     gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[2],
        #                                # lengthscale_prior = gpytorch.priors.GammaPrior(1,10),
        #                                batch_shape=torch.Size([self.num_out])),
        #     batch_shape=torch.Size([self.num_out]),
        #     outputscale_constraint=gpytorch.constraints.Interval(0.001, 0.001001),
        #     # outputscale_prior = gpytorch.priors.GammaPrior(1.5,2),
        # )
        self.task_covar_module = MultitaskKernel(self.covar_module, num_tasks=train_y.size(-1), rank=1)

    def forward(self, x):
        mean_x = self.mean_module(x).unsqueeze(-1).repeat(1, self.task_covar_module.num_tasks)
        covar_x = self.task_covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)

class PILCOAgent:
    def __init__(self, obs_space_dims, action_space_dims, lr=1e-3, gamma=0.95, batch_size=1000, device='cpu'):
        self.policy_network = PolicyNetwork(obs_space_dims, action_space_dims).to(device)
        self.optimizer_policy = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.likelihood = None
        self.gp_model = None

        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.next_states = []

        self.memo = 0

    def sample_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        mean, std = self.policy_network(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def create_gp_model(self, train_x, train_y):
        # 标准化训练数据
        train_x, self.train_x_mean, self.train_x_std = standardize(train_x)
        train_y, self.train_y_mean, self.train_y_std = standardize(train_y)

        if self.gp_model is None:
            self.train_x = train_x
            self.train_y = train_y
            self.likelihood = MultitaskGaussianLikelihood(num_tasks=train_y.size(-1)).to(self.device)
            self.gp_model = MultitaskGPModel(self.train_x, self.train_y, self.likelihood).to(self.device)
        else:
            self.train_x = train_x
            self.train_y = train_y
            self.gp_model.set_train_data(inputs=self.train_x, targets=self.train_y, strict=False)

    def train_gp_model(self, train_x:list, train_y:list, learning_rate=0.001, epochs=50):
        train_x = torch.FloatTensor(np.array(train_x)).to(self.device)
        train_y = torch.FloatTensor(np.array(train_y)).to(self.device)
        self.create_gp_model(train_x, train_y)
        self._train_gp_model(learning_rate = learning_rate, epochs = epochs)

    def _train_gp_model(self, learning_rate=0.004, epochs=80):
        self.gp_model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.gp_model.parameters(), lr=learning_rate)
        mll = ExactMarginalLogLikelihood(self.likelihood, self.gp_model).to(self.device)
        for i in range(epochs):
            optimizer.zero_grad()
            output = self.gp_model(self.train_x)
            loss = -mll(output, self.train_y)
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.gp_model.parameters(), 2.5)
            optimizer.step()
            if loss.item()<-2.5:
                break
        print('loss: ', loss.item())

    def predict_next_state(self, states, actions):
        self.gp_model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # 标准化输入数据
            state_action = torch.cat([states, actions], dim=-1).to(self.device)
            state_action = (state_action - self.train_x_mean) / self.train_x_std

            observed_pred = self.likelihood(self.gp_model(state_action))
            mean = observed_pred.mean

            # 解标准化预测结果
            mean = unstandardize(mean, self.train_y_mean, self.train_y_std)
        return mean

    def store_transition(self, state, action, reward, log_prob, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.next_states.append(next_state)
        self.memo += 1
        if self.memo == self.batch_size:
            self.update()

    def update(self):
        if self.memo == 0:
            return
        self.memo = 0
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(self.rewards):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        discounted_rewards = torch.FloatTensor(discounted_rewards).unsqueeze(1).to(self.device)  # 确保 discounted_rewards 的形状为 [batch_size, 1]

        log_probs = torch.stack(self.log_probs).to(self.device)
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).unsqueeze(1).to(self.device)  # 确保 actions 维度正确
        next_states = torch.FloatTensor(np.array(self.next_states)).to(self.device)

        # 预测状态差值 delta_states
        delta_states = next_states - states  # delta_states 的维度应该是 (batch_size, 6)

        # 确保 train_x 和 train_y 维度匹配
        train_x = torch.cat([states, actions], dim=-1)  # train_x 的维度是 (batch_size, 7)
        train_y = delta_states  # 预测 delta_states，维度是 (batch_size, 6)

        self.create_gp_model(train_x, train_y)
        self._train_gp_model()

        predicted_deltas = self.predict_next_state(states, actions).squeeze()
        predicted_next_states = states + predicted_deltas

        # 使用 current_reward 函数计算 predicted_next_states 的奖励
        predicted_rewards = torch.FloatTensor([current_reward(state) for state in predicted_next_states]).to(self.device)

        # 计算每个时间步的折扣因子
        discount_factors = torch.FloatTensor([self.gamma ** t for t in range(len(self.rewards))]).unsqueeze(1).to(self.device)

        advantages = discounted_rewards - (predicted_rewards.unsqueeze(1) * discount_factors)  # 确保 predicted_rewards 的形状为 [batch_size, 1]

        self.optimizer_policy.zero_grad()
        # 更新策略网络
        policy_loss = -torch.sum(log_probs * advantages.detach())
        policy_loss.backward()
        self.optimizer_policy.step()

        # 清除已使用的样本
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.next_states = []


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

def standardize(data):
    mean = data.mean(dim=0, keepdim=True)
    std = data.std(dim=0, keepdim=True) + 1e-8  # 加上一个小值防止除以0
    standardized_data = (data - mean) / std
    return standardized_data, mean, std

def unstandardize(data, mean, std):
    return data * std + mean

