import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 超参数
memory = 10000
batch_size = 32

class DDPGAgent:
    def __init__(self, no_of_states, no_of_actions, a_bound=20.0, lr_a=0.001, lr_c=0.002, gamma=0.9, alpha=0.01, device='cpu'):
        self.no_of_actions = no_of_actions
        self.no_of_states = no_of_states
        self.a_bound = a_bound

        self.lr_a = lr_a
        self.lr_c = lr_c
        self.gamma = gamma
        self.alpha = alpha
        self.device = torch.device(device)  # 将设备存储为torch.device对象

        self.memory = np.zeros((memory, no_of_states * 2 + no_of_actions + 1), dtype=np.float32)
        self.pointer = 0
        self.noise_variance = 0.03

        self.actor_eval = self.build_actor_network().to(self.device)
        self.actor_target = self.build_actor_network().to(self.device)
        self.critic_eval = self.build_critic_network().to(self.device)
        self.critic_target = self.build_critic_network().to(self.device)

        self.actor_optimizer = optim.Adam(self.actor_eval.parameters(), lr=self.lr_a)
        self.critic_optimizer = optim.Adam(self.critic_eval.parameters(), lr=self.lr_c)

        self.mse_loss = nn.MSELoss()

    def build_actor_network(self):
        return nn.Sequential(
            nn.Linear(self.no_of_states, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.no_of_actions),
            nn.Tanh()
        )

    def build_critic_network(self):
        return nn.Sequential(
            nn.Linear(self.no_of_states + self.no_of_actions, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def sample_action(self, state):
        return self.choose_action(state), 0

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # 确保输入的维度是 (1, no_of_states) 并移动到指定设备
        action = self.actor_eval(state).detach().cpu().numpy()[0]  # 通过网络前向传播获得动作，并移动到CPU
        action = np.clip(np.random.normal(action, self.noise_variance), -self.a_bound, self.a_bound)  # 添加噪声并裁剪
        # action = np.where((action > -2) & (action < 0), -2, action)
        # action = np.where((action < 2) & (action > 0), 2, action)
        return action

    def store_transition(self, s, a, r, s_):
        trans = np.hstack((s, a, [r], s_))
        index = self.pointer % memory
        self.memory[index, :] = trans
        self.pointer += 1

        if self.pointer > memory:
            self.noise_variance *= 0.99995
            self.learn()

    def soft_update(self, eval_net, target_net):
        for eval_param, target_param in zip(eval_net.parameters(), target_net.parameters()):
            target_param.data.copy_((1 - self.alpha) * target_param.data + self.alpha * eval_param.data)

    def learn(self):
        self.soft_update(self.actor_eval, self.actor_target)
        self.soft_update(self.critic_eval, self.critic_target)

        indices = np.random.choice(memory, size=batch_size)
        batch_transition = self.memory[indices, :]
        batch_states = torch.FloatTensor(batch_transition[:, :self.no_of_states]).to(self.device)
        batch_actions = torch.FloatTensor(batch_transition[:, self.no_of_states: self.no_of_states + self.no_of_actions]).to(self.device)
        batch_rewards = torch.FloatTensor(batch_transition[:, -self.no_of_states - 1: -self.no_of_states]).to(self.device)
        batch_next_states = torch.FloatTensor(batch_transition[:, -self.no_of_states:]).to(self.device)

        q_eval = self.critic_eval(torch.cat([batch_states, batch_actions], dim=1))
        next_actions = self.actor_target(batch_next_states)
        q_next = self.critic_target(torch.cat([batch_next_states, next_actions], dim=1)).detach()
        q_target = batch_rewards + self.gamma * q_next

        critic_loss = self.mse_loss(q_eval, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic_eval(torch.cat([batch_states, self.actor_eval(batch_states)], dim=1)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def save_model(self, path: str):
        torch.save({
            'actor_eval_state_dict': self.actor_eval.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_eval_state_dict': self.critic_eval.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'optimizer_actor_state_dict': self.actor_optimizer.state_dict(),
            'optimizer_critic_state_dict': self.critic_optimizer.state_dict()
        }, path)
        print(f'Successfully saved model to {path}.')

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_eval.load_state_dict(checkpoint['actor_eval_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_eval.load_state_dict(checkpoint['critic_eval_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['optimizer_critic_state_dict'])
        self.actor_eval.train()
        self.critic_eval.train()
        print(f'Successfully loaded model from {path}.')
