import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim

        # define an RNN with specified parameters
        # batch_first means that the first dim of the input and output will be the batch_size
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, bias=False, batch_first=True)

        # now we specify: NNxy
        self.fc = nn.Linear(hidden_dim, output_size, bias=False)

    def forward(self, u, hidden):
        # u (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)
        batch_size = u.size(0)

        # get RNN outputs
        r_out, hidden = self.rnn(u, hidden)
        # shape output to be (batch_size*seq_length, hidden_dim)
        r_out = r_out.view(-1, self.hidden_dim)
        # r_out: kth row is a hiden state vector at time k

        # get final output
        output = self.fc(r_out)
        return output, hidden

# let's simulate RNN with some input data
if __name__ == '__main__':
    input_size = 1
    output_size = 1
    hidden_dim = 3
    n_layers = 1

    myRNN = RNN(input_size,output_size,hidden_dim,n_layers)

    #generate input data for simulation
    seq_length = 20
    time_steps = np.linspace(0,np.pi,seq_length+1)
    data = np.cos(time_steps)
    data.resize((seq_length+1,1))
    u = data[:-1]   #all but the last piece of data
    u_tensor = torch.Tensor(u).unsqueeze(0)
    x0_tensor = torch.ones(1,1,3)

    y_rnn, xf = myRNN(u_tensor,x0_tensor)

    plt.figure()
    plt.plot(y_rnn.data.numpy())
    plt.show()

    # develop our own RNN using state space model

    # obtain the weights

    w_xx = myRNN.rnn.weight_hh_l0.data.numpy()
    w_xy = myRNN.fc.weight.data.numpy()
    w_ux = myRNN.rnn.weight_ih_l0.data.numpy()

    #begin our simulation

    # initialization

    uhat = np.mat(u.reshape(-1,seq_length))
    x = np.mat(np.zeros((hidden_dim,seq_length)))
    y = np.mat(np.zeros((output_size,seq_length)))

    xhat = x0_tensor.numpy().reshape(3,1)

    for k in range(seq_length):
        if k==0:
            x[:,k]=np.tanh(w_xx@xhat + w_ux@uhat[:,k])
            y[:,k]=w_xy@x[:,k]
        else:
            x[:,k]=np.tanh(w_xx@x[:,k-1] + w_ux@uhat[:,k])
            y[:,k]=w_xy@x[:,k]

    plt.figure()
    plt.plot(y[0,:].T)
    plt.show()