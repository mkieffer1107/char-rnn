# https://d2l.ai/chapter_recurrent-modern/lstm.html
# https://colah.github.io/posts/2015-08-Understanding-LSTMs
'''
Vanilla RNNs are prone to the vanishing and exploding gradient problems.
Models that handle longer sequence data are more susceptible to this, as
more grads have to be backpropagated through time (longer chains of multiplication). 
To handle the exploding gradient problem, we can use gradient clipping, assuming
that we are willing to lose some precision on parameter updates. However, there
is no simple way to handle vanishing gradients with the vanilla architecture.

Long short-term memory (LSTM) cells help solve this problem. Vanilla RNNs encode 
long term memory in their parameters during training. Short-term memory is 
captured in the hidden state of the RNN cell during inference. The LSTM cell
incorporates both time-length memories into a single cell for better performance.
In doing so, they are also able to maintain long-term dependencies, while also

The LSTM cell has two states that it keeps track of:

    - H_t: the hidden state at step t
    - C_t: the cell internal state at step t

At time step t, input X_t, previous hidden state H_{t-1}, and the previous
cell state C_{t-1} are passed into a cell to produce the new states H_t and C_t.

Suppose there are h hidden units, the batch size is n, and the number of inputs is d.
Then,

    - X_t: batched inputs have shape (n, d)
    - H: batched hidden states have shape (n, h)
    - C: cell internal state has shape (n, h)

The LSTM cell consists of a few gates that determine how to handle an input
at time step t, and how to update the hidden + cell states:

    - the forget gate layer, F_t
        - takes in H_{t-1} and X_t
        - sigmoid activation with range (0, 1)
        - determines whether to keep the current value of the memory or flush it
        - F_t = sig( X_t @ W_xf + H_{t-1} @ W_hf + b_f )

    - the input gate layer, I_t
        - takes in H_{t-1} and X_t
        - sigmoid activation with range (0, 1)
        - determines how much of the input node's value should be added to 
          the current memory cell internal state
        - I_t = sig( X_t @ W_xi + H_{t-1} @ W_hi + b_i )

    - input node, D_t = C^{~}_{t} (candidate cell state)
        - takes in H_{t-1} and X_t
        - tanh activation with range (-1, 1)
        - new candidate cell state to be added to the current cell state
        - D_t = tanh( X_t @ W_xc + H_{t-1} @ W_hc + b_c )

    - the output gate layer, O_t
        - takes in H_{t-1} and X_t
        - tanh activation with range (-1, 1)
        - determines whether the memory cell should influence the output 
          at the current time step
        - O_t = sig( X_t @ W_xo + H_{t-1} @ W_ho + b_o )

Where 

    - inputX-to-forget layer W_xf has shape (d, h)
    - inputX-to-input layer W_xi has shape (d, h)
    - inputX-to-output layer W_xo has shape (d, h)
    - inputX-to-cell state layer W_xc has shape (d, h)

    - hidden-to-forget layer W_hf has shape (h, h)
    - hidden-to-input layer W_hi has shape (h, h)
    - hidden-to-output layer W_ho has shape (h, h)
    - hidden-to-cell state layer W_hc has shape (h, h)
    
    - biases all have shape (1, h)
    

These are combined to update the cell state and hidden state output:

    - to update the internal cell state, the forget gate takes in the 
      the input value and hidden state, combines them, and then
      squeezes each value between 0 and 1

        - F_t = sig( X_t @ W_xf + H_{t-1} @ W_hf + b_f)

    - the input gate squeezes values between 0 and 1, while the input
      node squeezes values between -1 and 1

        - I_t = sig( X_t @ W_xi + H_{t-1} @ W_hi + b_i)

        - D_t = tanh( X_t @ W_xc + H_{t-1} @ W_hc + b_c)

    - the input and forget gate squeeze values between 0 and 1 because 
      they determine what fraction out of the whole the previous cell
      state values should be passed through to the next state
    
    - the input node values are between -1 and 1 because they are meant
      to be added to the previous cell state to update it
    
    - so the cell state update can be written as

        - C_t = F_t * C_{t-1} + I_t * D_t

    - where the input gate I_t governs how much we take new data into account via
      D_t and the forget gate F_t addresses how much of the old cell internal state
      C_{t-1} we retain, where * represents element-wise multiplication

    - the hidden state H_t is updated via the ouput gate, which squeezes values
      between 0 and 1
        
        - O_t = sig( X_t @ W_xo + H_{t-1} @ W_ho + b_o)

    - then tanh squeezes the cell state C_t values between -1 and 1, and is
      then element-wise multiplied by O_t, so H_t is always in range (-1, 1)

        - H_t = O_t * tanh(C_t)
      
If all of the values of the forget gate are saturated to 1, and the values of the
input gates are saturated to 0, then the cell state does not change

    - C_t = F_t * C_{t-1} + I_t * D_t = 1 * C_{t-1} + 0 * D_t = C_{t-1}

    - H_t = O_t * tanh(C_t) = O_t * tanh(C_{t-1})

and we recover the vanilla RNN (special state of LSTM cell) where only the
hidden state H_t changes with time.

When the output gate values are saturated at 1, the internal memory state of
the cell impacts the subsequent layers uninhibited

     - H_t = O_t * tanh(C_t) = 1 * tanh(C_t) = tanh(C_t)

However, when the output gate values are saturated at 0, the current memory is 
prevented from impacting other layers of the network at the current time step

    - H_t = O_t * tanh(C_t) = 0 * tanh(C_t) = 0

Note that a memory cell can accrue information across many time steps without 
impacting the rest of the network (as long as the output gate takes values close 
to 0), and then suddenly impact the network at a subsequent time step as soon as 
the output gate flips from values close to 0 to values close to 1
'''

import torch
import torch.nn as nn

class LSTM(nn.Module):
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
        self.W_xi, self.W_hi, self.b_i = triple()  # input gate
        self.W_xf, self.W_hf, self.b_f = triple()  # forget gate
        self.W_xo, self.W_ho, self.b_o = triple()  # output gate
        self.W_xc, self.W_hc, self.b_c = triple()  # input node
        self.dropout = nn.Dropout(dropout)         # dropout layer

    def to(self, device):
        super().to(device)
        self.device = device
        return self
    
    def forward(self, X, H_C=None):
        # X has shape (seq_length, batch_size, input_size)
        # H_C is a tuple containing the cell internal state + hidden state: (H, C)
        #   H and C have shape (batch_size, hidden_size)
        if H_C is None:
            H = torch.zeros((X.shape[1], self.hidden_size), device=X.device)
            C = torch.zeros((X.shape[1], self.hidden_size), device=X.device)
        else:
            H, C = H_C

        outputs = []
        for x_t in X:
            # calculate gate values and internal node for each batch
            # x_t has shape (batch_size, input_size)
            I = torch.sigmoid(x_t @ self.W_xi + H @ self.W_hi + self.b_i)
            F = torch.sigmoid(x_t @ self.W_xf + H @ self.W_hf + self.b_f)
            O = torch.sigmoid(x_t @ self.W_xo + H @ self.W_ho + self.b_o)
            C_tilde = torch.tanh(x_t @ self.W_xc + H @ self.W_hc + self.b_c)
            
            # update the states
            C = F*C + I*C_tilde   # C_t = ( F_t * C_{t-1} ) + ( I_t * C~_t )
            H = O * torch.tanh(C) # H_t = O_t * tanh(C_t)
            
            # apply dropout to the hidden state / output -- not to internal cell state
            H = self.dropout(H)
            outputs.append(H)
        
        # convert list of tensors to a single tensor with shape (seq_length, batch_size, hidden_size)
        outputs = torch.stack(outputs, dim=0)
        return outputs, (H, C)