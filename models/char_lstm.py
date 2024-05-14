'''
RNN created using PyTorch for decoding / character-level
language modeling

# https://youtu.be/yCC09vCHzF8?si=Eex7cHXV6cJ_IUP_
# https://d2l.ai/chapter_recurrent-neural-networks/bptt.html

streaming
learn how to make the output write as soon as it is calculated

'''
import torch
import torch.nn as nn
from layers import LSTM

class CharLSTM(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, seq_length):
        super.__init__()
        self.seq_length = seq_length

        # create the RNN layer
        self.rnn = LSTM()

        # output at time t depends on hidden state t
        # O_t = H_t @ W_ho + b_o
        self.W_ho  = nn.Parameter(
            torch.randn(hidden_size, output_size) * 0.01
        )
        self.b_o = nn.Parameter(torch.zeros(output_size)) # output bias

    def forward(self, x):
        ...


# class CharLSTM(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, device="cpu"):
#         super(CharLSTM, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.lstm = CustomLSTM(embedding_dim, hidden_dim, device=device)
#         self.fc = nn.Linear(hidden_dim, vocab_size)
    
#     def forward(self, x, hidden):
#         x = self.embedding(x)
#         out, hidden = self.lstm(x, hidden)
#         out = self.fc(out[-1])  # Get the last output
#         return out, hidden
    
#     def init_hidden(self, batch_size):
#         return (torch.zeros(batch_size, self.lstm.hidden_size).to(self.lstm.device),
#                 torch.zeros(batch_size, self.lstm.hidden_size).to(self.lstm.device))
