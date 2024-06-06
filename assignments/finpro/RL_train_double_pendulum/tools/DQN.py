from collections import namedtuple, deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 2000
EPISODE=2000

########### from pytorch ###############

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# class Net(nn.Module):
#     def __init__(self, n_state, n_hidden, n_action, device = 'cpu'):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(n_state, n_hidden)
#         self.fc1.weight.data.normal_(0, 0.1)   # initialization
#         self.out = nn.Linear(n_hidden, n_action)
#         self.out.weight.data.normal_(0, 0.1)   # initialization
#         self.device = device
#
#     def forward(self, x):
#         x.to(self.device)
#         x = self.fc1(x)
#         x = F.relu(x)
#         actions_value = self.out(x)
#         return actions_value
#
#
# class DQN:
#     def __init__(self, net, target_net, n_state, n_hidden, n_action, device = 'cpu', lr=LR):
#         self.net = net
#         self.target_net = target_net
#         self.n_state = n_state
#         self.n_hidden = n_hidden
#         self.n_action = n_action
#         self.learn_step_counter = 0
#         self.memory_counter = 0
#         self.memory = np.zeros((MEMORY_CAPACITY, self.n_state * 2 + 2))  # initialize memory
#         self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
#         self.loss_func = nn.MSELoss()
#
#     def choose_action(self, x):
#         x = torch.unsqueeze(torch.FloatTensor(x), 0)
#         # input only one sample
#         if np.random.uniform() < EPSILON:
#             actions_value = self.net.forward(x)
#             action = torch.max(actions_value, 1)[1].data.cpu().numpy()
#             action = action[0]
#         else:
#             action = np.random.randint(0, self.n_action)
#         return action
#
#     def store_transition(self, s, a, r, s_):
#         transition = np.hstack((s, [a, r], s_))
#         # replace the old memory with new memory
#         index = self.memory_counter % MEMORY_CAPACITY
#         self.memory[index, :] = transition
#         self.memory_counter += 1
#
#     def learn(self):
#         # target parameter update
#         if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
#             self.target_net.load_state_dict(self.net.state_dict())
#         self.learn_step_counter += 1
#
#         # sample batch transitions
#         sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
#         batch_memory = self.memory[sample_index, :]
#         batch_s = torch.FloatTensor(batch_memory[:, :self.n_state])
#         batch_a = torch.LongTensor(batch_memory[:, self.n_state:self.n_state + 1].astype(int))
#         batch_r = torch.FloatTensor(batch_memory[:, self.n_state+ 1:self.n_state + 2])
#         batch_s_ = torch.FloatTensor(batch_memory[:, -self.n_state:])
#
#         q = self.net(batch_s).gather(1, batch_a)  # shape (batch, 1)
#         q_target = self.target_net(batch_s_).detach()  # detach from graph, don't backpropagate
#         y = batch_r + GAMMA * q_target.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
#         loss = self.loss_func(q, y)
#
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()