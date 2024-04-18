# import numpy as np
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

#static assignments
input_size = 1
output_size = 1
hidden_size = 3
n_layers = 1
seq_length = 25

class MyRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers):
        super(MyRNN, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.rnn = nn.RNN(input_size, hidden_size, n_layers, bias = False, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        # set bias to False
        self.fc.bias.data.fill_(0)
    def forward(self, x, hidden):
        batch_size = x.size(0)

        r_out, hidden = self.rnn(x, hidden)
        r_out = r_out.view(-1, self.hidden_size)
        output = self.fc(r_out)

        return output, hidden

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        return hidden


mnn = MyRNN(input_size, output_size, hidden_size, n_layers)
time_steps = np.linspace(0, np.pi, seq_length + 1)
data =np.sin(time_steps)
data.resize((seq_length+1, 1))

u = data[:-1]

u_tensor = torch.Tensor(u).unsqueeze(0)
x0_tensor = torch.ones(1,1,3)


if __name__ == '__main__':
    print('u_tensor:', u_tensor)
    print('x0_tensor:', x0_tensor)

    y,xt = mnn(u_tensor, x0_tensor)

    plt.figure()
    plt.plot(y.data.numpy())
    plt.show()

    data_rnn = y.data.numpy()


    # initialize the hidden state
    # setup the dimensions of the hidden state
    u = np.mat(u.reshape(-1, seq_length))
    x = np.mat(np.zeros((hidden_size, seq_length)))
    y = np.mat(np.zeros((output_size, seq_length)))

    w_xx = mnn.rnn.weight_hh_l0.data.numpy()
    w_xy = mnn.fc.weight.data.numpy()
    w_ux = mnn.rnn.weight_ih_l0.data.numpy()

    # begin to simulate
    # uhat = np.mat(u.reshape(-1, seq_length))
    xhat = x0_tensor.numpy().reshape(hidden_size, 1)
    # xhat = np.mat(u.reshape(-1,seq_length))
    for k in range(seq_length):
        if k == 0:
            x[:, k] = np.tanh(w_xx@xhat+w_ux@u[:,k])
            y[:, k] = w_xy@x[:,k]
        else:
            x[:, k] = np.tanh(w_xx@x[:,k-1]+w_ux@u[:,k])
            y[:,k] = w_xy@x[:,k]
        # if k == 0:
        #     x[:, k] = np.tanh(w_xx @ xhat + w_ux @ uhat[:, k])
        #     y[:, k] = w_xy @ x[:, k]
        # else:
        #     x[:, k] = np.tanh(w_xx @ x[:, k - 1] + w_ux @ uhat[:, k])
        #     y[:, k] = w_xy @ x[:, k]

    plt.plot(y.T)
    plt.show()

    plt.figure()
    plt.plot(data_rnn)
    plt.plot(y.T)
    plt.legend(['RNN','mRNN'])
    plt.show()