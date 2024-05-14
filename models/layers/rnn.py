# https://d2l.ai/chapter_recurrent-modern/deep-rnn.html
# https://d2l.ai/chapter_recurrent-neural-networks/rnn-scratch.html
# https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
# https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/rnn.py
# https://github.com/tinygrad/tinygrad/blob/master/models/rnnt.py
'''
Suppose we have batches of data X_t with shape (n, d), where n is the number of samples
with d dimensional inputs. Let h be the number of hidden units in the hidden layer and
d' be the number of outputs.

X_t has shape (n, d) = (batch_size, input_size)
H_t has shape (n, h) = (batch_size, hidden_size)
O_t has shape (n, d') = (batch_size, output_size)


We have three weight matrices:
    W_ih: input to hidden,  (input_size, hidden_size)
    W_hh: hidden to hidden, (hidden_size, hidden_size)
    W_ho: hidden to output, (hidden_size, output_size)

And biases:
    b_h: hidden bias, (hidden_size)
    b_o: output bias, (output_size)

A non-linear activation function given by phi

To update the hidden state given previous state H_{t-1}

    H_t = phi(X_t @ W_ih + H_{t-1} @ W_hh  + b_h)

        = (n,d)@(d,h) + (n,h)@(h,h) + (h) = (n,h) + (n,h) + (n,h) = (n,h)

This can actually be performed by concatenating (X_t and H_{t-1}) and (W_ih and W_hh), 
and multiplying the resulting matrices. This can lead to some speedup, but I won't 
use it here.

To get the output

    O_t = H_t @ W_ho + b_o

        = (n,h)@(h,d') = (n,d')

However, this is just one RNN layer (input and hidden state), so output is not accessed here.
To access the output, incorporate the RNN layer as a layer in a neural network.
'''

# common RNN layer widths (h) are in the range (64, 2056), and common depths (L) are in the range (1, 8)
import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, device, sigma=0.01):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size

        # ih means input to hidden
        # (X @ W_ih) == (n,d)@(d,h) = (n,h)
        self.W_ih = nn.Parameter(
            torch.randn(input_size, hidden_size, device = self.device) * sigma,
        )
        
        # hh means hidden to hidden
        # (H_{t-1} @ W_hh + b_h) == (h,h)@(h,h) + (h,) = (h,h)
        self.W_hh = nn.Parameter(
            torch.randn(hidden_size, hidden_size, device = self.device) * sigma,
        )

        self.b_h = nn.Parameter(
            torch.zeros(hidden_size, device = self.device)
        ) # hidden bias

        # NOTE: could also use nn.Linear layers here instead of explicit matrices

    def to(self, device):
        super().to(device)
        self.device = device
        return self
    
    def forward(self, X, H=None):
        # X has shape (seq_length, batch_size, input_size)
        # hidden states, H_t, where index is time step t
        #   shape: (batch_size, hidden_size)
        # no need to embed input samples for binary values (0 or 1)
        if H is None:
            batch_size = X.shape[1]
            H = torch.zeros((batch_size, self.hidden_size), device=self.device)

        # store the hidden state at each timestep
        #   sometimes called outputs -- meaning cells states
        states = [] 

        # iterate over each sequence (time step, t)
        for t in range(X.shape[0]):
            # takes all samples in batch for sequence t
            # x_t has shape (batch_size, input_size) -- inputs at timestep t
            x_t = X[t, :, :]  
            #  could do cat(x_t, H) @ cat(W_ih, W_hh)
            H = torch.tanh(x_t @ self.W_ih + H @ self.W_hh + self.b_h)
            states.append(H)

        # H has shape (batch_size, hidden_size)
        return states, H