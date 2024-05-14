# https://d2l.ai/chapter_recurrent-modern/gru.html
# https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be
'''
Gated Recurrent Units (GRUs) are simpler and quicker than LSTMs. In contrast to an LSTM cell,
a GRU cell only has two gates (reset + update).

The GRU cell has one state that it keeps track of:

    - H_t: the hidden state at step t

At time step t, input X_t and the previous hidden state H_{t-1} are passed into 
a cell to produce the new state H_t.

Suppose there are h hidden units, the batch size is n, and the number of inputs is d.
Then,

    - X_t: batched inputs have shape (n, d)
    - H: batched hidden states have shape (n, h)

The GRU cell consists of a few layers that determine how to handle an input
at time step t, and how to update the hidden + cell states:

    - reset gate layer, R_t
        - takes in H_{t-1} and X_t
        - sigmoid activation with range (0, 1)
        - controls how much of the previous state to remember 
        - R_t = sig( X_t @ W_xr + H_{t-1} @ W_hr + b_r )
    
    - update gate, Z_t
        - takes in H_{t-1} and X_t
        - sigmoid activation with range (0, 1)
        - controls how much of the previous state to forget
        - Z_t = sig( X_t @ W_xz + H_{t-1} @ W_hz + b_z )

    - candidate hidden state, H^{~}_t = G_t
        - takes in H_{t-1} and X_t
        - tanh activation with range (-1, 1)
        - new candidate hidden state to be added to the current hidden state
        - G_t = tanh( X_t @ W_xh + ( R_t * H_{t-1} ) @ W_hh + b_h )

Where 

    - input-to-reset layer W_xr has shape (d, h)
    - input-to-update layer W_xz has shape (d, h)
    - input-to-hidden layer W_xh has shape (d, h)
  
    - hidden-to-reset layer W_hr has shape (h, h)
    - hidden-to-update state layer W_hz has shape (h, h)
    - hidden-to-hidden state layer W_hh has shape (h, h)
    
    - biases all have shape (1, h)

These are combined to update the hidden state output:

    - the update gate takes in the the input value and hidden state, 
      combines them, and then squeezes each value between 0 and 1

        - Z_t = sig( X_t @ W_xz + H_{t-1} @ W_hz + b_z )

    - the reset gate squeezes values between 0 and 1, while the candidate
      hidden state squeezes values between -1 and 1

        - R_t = sig( X_t @ W_xr + H_{t-1} @ W_hr + b_r )

        - G_t = tanh( X_t @ W_xh + ( R_t * H_{t-1} ) @ W_hh + b_h )

    - the update and reset gate squeeze values between 0 and 1 because 
      they determine what fraction out of the whole the previous hidden
      state values should be passed through to the next state
    
    - the candidate hidden state values are between -1 and 1 because they are meant
      to be added to the previous hiddenf state to update it
    
    - so the hidden state update can be written as a convex combination of H_t and G_t = H^{~}_t

        - H_t = Z_t * H_{t-1} + (1 - Z_t) * G_t

If all of the values in the reset gate are saturated to 1, then the candidate hidden state
equation recovers the vanilla RNN hidden state update equation

    - G_t = tanh( X_t @ W_xh + ( R_t * H_{t-1} ) @ W_hh + b_h ) 
          = tanh( X_t @ W_xh + ( 1 * H_{t-1} ) @ W_hh + b_h )
          = tanh( X_t @ W_xh + H_{t-1} @ W_hh + b_h )

If the values in the reset gate are saturated to 0, then we recover an MLP layer

    - G_t = tanh( X_t @ W_xh + ( R_t * H_{t-1} ) @ W_hh + b_h ) 
          = tanh( X_t @ W_xh + ( 0 * H_{t-1} ) @ W_hh + b_h )
          = tanh( X_t @ W_xh + b_h )

If the values in the update gate are saturated to 1, then the hidden state does not change, and
we retain the old state
    
    - H_t = Z_t * H_{t-1} + (1 - Z_t) * G_t  = 1 * H_{t-1} + (1 - 1) * G_t = H_{t-1}

which means that we ignore the information from X_t, effectively skipping time step t.

If the values in the update gate are saturated to 0, then we replace the previous hidden
state with the candidate latent state G_t
    
    - H_t = Z_t * H_{t-1} + (1 - Z_t) * G_t  = 0 * H_{t-1} + (1 - 0) * G_t = G_t
'''

import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, device="cpu", dropout=0.1, sigma=0.01):
        super().__init__()
        self.dropout = dropout

        # create weights from given shape
        init_weight = lambda *shape: \
            nn.Parameter(torch.randn(*shape, device=device) * sigma)
        
        # layers: (inputX-to-gate (d, h), hidden-to-gate (h, h), bias (1, h))
        triple = lambda: (init_weight(input_size, hidden_size),
                          init_weight(hidden_size, hidden_size),
                          nn.Parameter(torch.zeros(hidden_size, device=device)))
        
        # unpack tuples and initialize layers
        self.W_xz, self.W_hz, self.b_z = triple()  # update gate
        self.W_xr, self.W_hr, self.b_r = triple()  # reset gate
        self.W_xh, self.W_hh, self.b_h = triple()  # candidate hidden state
        self.dropout = nn.Dropout(dropout)         # dropout layer
    
    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def forward(self, X, H=None):
        # X has shape (seq_length, batch_size, input_size)
        # H has shape (batch_size, hidden_size)
        if H is None:
            H = torch.zeros((X.shape[1], self.num_hiddens), device=X.device)

        outputs = []
        for x_t in X:
            # calculate gate values and internal node for each batch
            # x_t has shape (batch_size, input_size)
            Z = torch.sigmoid(torch.matmul(x_t, self.W_xz) + torch.matmul(H, self.W_hz) + self.b_z)
            R = torch.sigmoid(torch.matmul(x_t, self.W_xr) + torch.matmul(H, self.W_hr) + self.b_r)
            H_tilde = torch.tanh(torch.matmul(x_t, self.W_xh) + torch.matmul(R * H, self.W_hh) + self.b_h)
            
            # update the hidden state
            H = Z * H + (1 - Z) * H_tilde
            
            # apply dropout to the hidden state
            H = self.dropout(H)
            outputs.append(H)
        return outputs, H