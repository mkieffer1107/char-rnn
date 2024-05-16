'''
RNN created using PyTorch for decoding / character-level
language modeling

# https://youtu.be/yCC09vCHzF8?si=Eex7cHXV6cJ_IUP_
# https://d2l.ai/chapter_recurrent-neural-networks/bptt.html

'''
import torch
import torch.nn as nn
from layers import LSTM

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, seq_length, device="cpu"):
        super(CharLSTM).__init__()
        self.seq_length = seq_length
        
        # create the embedding layer --> maps vocab to embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # create the RNN layer
        self.lstm = LSTM(embedding_dim, hidden_dim, num_layers, device=device)

        # output at time t depends on hidden state t
        # O_t = H_t @ W_ho + b_o
        self.fc = nn.Linear(hidden_dim, vocab_size)  

        # simpler to just use a linear layer ^^^
        # self.W_ho  = nn.Parameter(
        #     torch.randn(hidden_dim, vocab_size) * 0.01
        # )
        # self.b_o = nn.Parameter(torch.zeros(vocab_size)) # output bias

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = out[-1] # get the last output
        # out = out @ self.W_ho + self.b_o
        out = self.fc(out)
        return out, hidden
